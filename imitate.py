import os

from rlbench.observation_config import ObservationConfig

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model

from utils.network_info import NetworkInfo
from utils.utils import get_order
from utils.utils import load_data
from utils.utils import format_data
from utils.utils import format_time
from utils.utils import split_data

from evaluate import evaluate_network
from config import EndToEndConfig

from psutil import virtual_memory
from typing import List
from typing import Dict
from typing import Union
from os.path import join
from os import listdir
from os import getpid

import numpy as np
import pickle
import time
import gc

from multiprocessing import Queue
from queue import Empty
from multiprocessing import Process


def episode_loader(train_queue: Queue, episode_queue: Queue, network_info: NetworkInfo, obs_config: ObservationConfig):
    while True:
        if episode_queue.qsize() < 5:
            if train_queue.empty():
                # print(f'{getpid()}: Reached last element of training queue')
                break
            else:
                episode_num = train_queue.get()
                start = time.perf_counter()
                episode_queue.put(split_data(format_data(load_data(network_info.train_dir,
                                                                 episode_num,
                                                                 obs_config),
                                                       pov=network_info.pov
                                                       ),
                                           num_images=network_info.num_images,
                                           pov=network_info.pov
                                           )
                                )
                # print(f'{getpid()}: data for {episode_num} in queue took {time.perf_counter() - start} seconds')
        time.sleep(0.05)


def train_new(config: EndToEndConfig) -> None:
    print(f'\n[Info] Training a new model')

    train_dir, test_dir = config.get_train_test_directories()
    training_info = config.get_episode_amounts(train_dir, test_dir)

    network, network_info = config.get_new_network(training_info=training_info)

    train(network=network,
          network_info=network_info,
          save_root=config.network_root)


def train_existing(config: EndToEndConfig) -> None:
    print(f'\n[Info] Continuing the training of an existing model')

    ###############################################################
    # Get users selection and load it and its network information #
    ###############################################################

    network_dir = config.get_trained_network()

    with open(join(network_dir, 'network_info.pickle'), 'rb') as f:
        network_info = pickle.load(f)

    try:
        prev_train_performance = np.loadtxt(join(network_dir, 'train_performance.csv'),
                                            delimiter=",")
    except FileNotFoundError:
        prev_train_performance = None

    print('\n[Info] Retraining will not perform any evaluation. Use evaluate.py instead.')
    network_info.test_amount = 0  # Ensure that this is zero.

    print(f'\n[Info] Retraining will use {network_info.train_amount} episodes per epoch ')

    print(f'\n[Info] Retraining will use episodes from {network_info.train_dir}')

    #####################################
    # Get the parameters for retraining #
    #####################################

    request = int(input(f'\nHow many more epoch (at {network_info.total_epochs} currently)'
                        f' should the network be trained on (default is 1)? ') or 1)
    network_info.epochs_to_train = 1 if request < 1 else request
    print(f'Training the network for {network_info.epochs_to_train} additional epoch. ')

    ep_in_train_dir = len(listdir(network_info.train_dir))
    if network_info.train_amount != ep_in_train_dir:
        retraining_warning(network_info.train_amount, ep_in_train_dir)

    network = load_model(join(network_dir, network_dir.split('/')[-1] + '.h5'))

    train(network=network,
          network_info=network_info,
          save_root=config.network_root,
          prev_train_performance=prev_train_performance)


def train(network: Model,
          network_info: NetworkInfo,
          save_root: str,
          prev_train_performance: np.ndarray = None):
    ##################################################################################################
    # Update network info before training and generate the new it will be saved as and the directory #
    ##################################################################################################
    network_info.prev_epochs = network_info.total_epochs
    network_info.total_epochs += network_info.epochs_to_train

    save_network_as = f'{network_info.network_name}_{network_info.train_amount}_by{network_info.total_epochs}'
    network_save_dir = join(save_root, 'imitation', save_network_as)
    check_if_network_exists(network_save_dir)

    #####################################
    # RLBench settings for loading data #
    #####################################
    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True

    #########################################################
    # Information used by TensorFlow / in the training loop #
    #########################################################
    train_order = get_order(network_info.train_amount,
                            network_info.train_available,
                            network_info.epochs_to_train)
    total_episodes = len(train_order) - 1
    episode_count = 0

    steps = 0  # Counter for number of steps in each episode

    display_every = int(np.ceil((total_episodes + 1)/100))

    train_performance = []

    prev_last_step = prev_train_performance[-1, -1] if prev_train_performance is not None else 0

    # How many episodes should the network see before back propagation
    episodes_per_update = 2

    memory_percent_threshold = 70

    train_queue = Queue()
    episode_queue = Queue()

    for ep in train_order:
        train_queue.put(ep)

    print(f'Train queue size = {train_queue.qsize()} == {len(train_order)}')

    num_loaders = 2
    proc = []
    for loader in range(num_loaders):
        proc.append(Process(target=episode_loader,
                            args=(train_queue, episode_queue, network_info, obs_config)))
    [p.start() for p in proc]
    print('Started loader processes')

    #################
    # Training loop #
    #################
    print(f'\n[Info] Pre-training summary: ')
    print(f'Will train with {network_info.train_amount} episodes over {network_info.epochs_to_train} epochs. \n'
          f'Training episodes will be pulled from: {network_info.train_dir}\n'
          f'The network will be saved at: {network_save_dir}')

    input('\nReady to begin training. Press enter to proceed...')

    start_time = time.perf_counter()
    for i in range(len(train_order)):
        start = time.perf_counter()
        try:
            # print(f'getting data.. {episode_queue.qsize()}')
            inputs, labels = episode_queue.get()
            # print('got data')
            h = network.fit(x=inputs,
                            y=labels,
                            verbose=0,
                            shuffle=False,
                            epochs=1,  # Epochs are already handled by train_order
                            workers=os.cpu_count(),
                            use_multiprocessing=True,)
            # print(f'Loop {i}: took {time.perf_counter() - start} seconds.')
        except Empty:
            # print('waiting for data')
            time.sleep(0.1)

    free_memory()

    end_time = time.perf_counter()
    training_time = format_time(end_time - start_time)

    print(f'[Info] Finished training model. Training took {training_time}.')

    [p.join() for p in proc]

    ####################
    # Save the network #
    ####################
    network.save(join(network_save_dir, save_network_as + '.h5'))

    #####################################################################
    # Save network info, training performance, and a graph of the model #
    #####################################################################
    save_info_at = join(network_save_dir, 'network_info.pickle')
    with open(save_info_at, 'wb') as handle:
        pickle.dump(network_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_train_performance(network_save_dir=network_save_dir,
                           train_performance=train_performance,
                           prev_train_performance=prev_train_performance)

    try:
        plot_model(network, join(network_save_dir, "network.png"), show_shapes=True)
    except ImportError:
        print(f"\n[Warn] Could not print network image. You must install pydot (pip install pydot) "
              f"and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for "
              f"plot_model/model_to_dot to work")

    ######################################
    # Optionally do some testing testing #
    ######################################
    if network_info.test_amount > 0:
        evaluate_network(network=network,
                         network_info=network_info,
                         network_save_dir=network_save_dir,
                         obs_config=obs_config)

    print(f'\n[Info] Successfully exiting program.')


def save_train_performance(network_save_dir: str,
                           train_performance: List[Dict],
                           prev_train_performance: np.ndarray = None) -> None:
    """
    Saves the training performance as a CSV file. Assumes each output has at most one loss
    and one metric. History is saved as train_performance.csv in the network directory.

    :param network_save_dir:       Root directory for the network the data is associated with
    :param train_performance:      Performance from the most recent round of training
    :param prev_train_performance: If retraining this is a array with the data from previous
                                   training that has been loaded from memory.

    """
    train_array = []
    for step in train_performance:
        data = np.array(list(step.items()), dtype='object')

        # Need to reshape since TF returns losses/metrics as a list
        # we assume there is only one element. Using multiple metrics
        # or losses on one output will break this step.
        array = np.array(data[:, 1].tolist()).astype('float')
        array = np.reshape(array, array.shape[0])

        train_array.append(array)

    train_array = np.array(train_array)

    if prev_train_performance is not None:
        train_array = np.vstack((prev_train_performance, train_array))

    header = ['loss', 'joint MSE loss', 'action sparse entropy loss', 'target MSE loss',
              'gripper MSE loss', 'joint RMS', 'action accuracy', 'target RMS', 'gripper RMS', 'steps']
    np.savetxt(join(network_save_dir, 'train_performance.csv'),
               train_array,
               delimiter=",",
               header=', '.join(header))


def display_update(network_info: NetworkInfo, episode_count: int, start_time: float, total_episodes: int) -> None:
    """
    Prints to screen an update on the current training status.

    :param network_info:   NetworkInfo object for the trained network
    :param episode_count:  Total number of episodes (including repeats) that have been loaded from memory for training
    :param start_time:     Time at which the network training began
    :param total_episodes: Total number of episodes (including repeats) that will be loaded from memory for training
    """
    print(f'[Info] {episode_count / (total_episodes + 1) * 100:3.1f}% Complete '
          f'{format_time((time.perf_counter() - start_time) * (total_episodes - episode_count + 1) / episode_count)} '
          f'remaining. Trained through episode '
          f'{episode_count - network_info.train_amount * int((episode_count - 1) / network_info.train_amount)} '
          f'of {network_info.train_amount} in epoch '
          f'{int((episode_count - 1) / network_info.train_amount) + 1 + network_info.prev_epochs} '
          f'of {network_info.total_epochs}.\n')


def free_memory(threshold: Union[int, None] = None) -> None:
    """
    Prevents memory overflow due to the custom training loop by explicitly collecting
    garbage in Python and clearing the TensorFlow backend.

    See: see: https://github.com/tensorflow/tensorflow/issues/37505
    """
    clear_session()  # Resolves memory overflow
    gc.collect()  # Explicitly called for safety. Does not affect speed.

    msg = '[Warn] '
    msg += f'System detected that over {threshold}% memory is in use. ' if threshold else ''
    msg += 'Called functions to clear free unused memory. '
    msg += 'Training will proceed.' if threshold else ''

    print(msg)


def retraining_warning(network_amount: int, ep_in_train_dir: int) -> None:
    """
    Called to warn that there is a different number of episodes in the training directory
    than the network was previously trained with.

    :param network_amount:  Number of episodes used by the network (usually from its NetworkInfo object)
    :param ep_in_train_dir: Number of episodes in the training directory
    """
    input(f'\n[WARN] It seems that there are {ep_in_train_dir} episodes in the training directory '
          f'and the network was trained on {network_amount} episodes previously. Using a '
          f'dataset with different (new or removed) episodes than the one the network was previously trained with '
          f'means the network will not see each episode equally. Each epoch may also consist of different episodes '
          f'than the prior one. Doing this makes it difficult to quantify the training regime.\n'
          f'If you still with to train in this manner, press enter to acknowledge the risk and continue... ')


def check_if_network_exists(network_save_dir: str) -> None:
    """
    Checks to see if the requested network will override any existing networks and asks
    for confirmation if it will.

    :param network_save_dir: Name of directory that the network info will be saved in
    """
    try:
        listdir(network_save_dir)
        print(f'\n[WARN] There is already a network at {network_save_dir} training will override this.')
        choice = input('Type "yes" to override the existing network (default is "no"): ')
        if choice != "yes":
            exit('\nWill not override existing network. Exiting program.')
    except FileNotFoundError:
        pass


def main():
    config = EndToEndConfig()

    prompt = '\nImitation training options: \n' \
             '0......To train a new model (default) \n' \
             '1......To continue training an existing model \n' \
             'Enter choice: '
    choice = int(input(prompt) or 0)

    if choice == 0:
        train_new(config)
    else:
        train_existing(config)


if __name__ == "__main__":
    main()
