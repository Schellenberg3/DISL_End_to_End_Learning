import os

from rlbench.observation_config import ObservationConfig
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from utils.utils import get_order
from utils.utils import load_data
from utils.utils import format_data
from utils.utils import check_yes
from utils.utils import format_time
from utils.utils import split_data
from utils.training_info import TrainingInfo
from utils.network_info import NetworkInfo
from tensorflow.keras import Model
from os.path import join
from os import listdir
from typing import Tuple
from typing import List
from typing import Dict
from typing import Union
from config import EndToEndConfig
from psutil import virtual_memory
import numpy as np
import tensorflow as tf
import gc
import pickle
import datetime
import time


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

    network_dir, network_name = config.get_trained_network()

    with open(join(network_dir, 'network_info.pickle'), 'rb') as f:
        network_info = pickle.load(f)

    print('\n[Info] Retraining will not perform any evaluation. Use evaluate.py instead.')
    network_info.test_amount = 0  # Ensure that this is zero.

    print(f'\n[Info] Retraining will use {network_info.train_amount} episodes per epoch ')

    print(f'\n[Info] Retraining will use episodes from {network_info.train_dir}')

    request = int(input(f'\nHow many more epoch (at {network_info.total_epochs} currently)'
                        f' should the network be trained on (default is 1)? ') or 1)
    network_info.epochs_to_train = 1 if request < 1 else request
    print(f'Training the network for {network_info.epochs_to_train} additional epoch. ')

    ep_in_train_dir = len(os.listdir(network_info.train_dir))
    if network_info.train_amount != ep_in_train_dir:
        retraining_warning(network_info.train_amount, ep_in_train_dir)

    network = load_model(network_dir)

    # todo write method to save and load this info
    prev_train_performance = None

    train(network=network,
          network_info=network_info,
          save_root=config.network_root,
          prev_train_performance=prev_train_performance)


def train(network: Model,
          network_info: NetworkInfo,
          save_root: str,
          prev_train_performance: np.ndarray = None):

    #####################################################
    # Get information related to the dataset / training #
    #####################################################


    ##########################################
    # Get information related to the network #
    ##########################################
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

    # How many episodes should the network see before back propagation
    episodes_per_update = 2

    checkpoint_callback = ModelCheckpoint(filepath='',
                                          save_weights_only=False,
                                          save_freq=100_000_000)

    memory_percent_threshold = 70

    #################
    # Training loop #
    #################
    print(f'\n[Info] Pre-training summary: ')
    print(f'Will train with {network_info.train_amount} episodes over {network_info.epochs_to_train} epochs. \n'
          f'Training episodes will be pulled from: {network_info.train_dir}\n'
          f'The network will be saved at: {network_save_dir}')

    input('\nReady to begin training. Press enter to proceed...')

    start_time = time.perf_counter()

    while episode_count <= total_episodes:
        train_angles = []
        train_action = []
        train_images = []

        label_angles = []
        label_action = []
        label_target = []
        label_gripper = []

        for episode in range(episodes_per_update):
            try:
                inputs, labels = split_data(format_data(load_data(network_info.train_dir,
                                                                  train_order[episode_count],
                                                                  obs_config),
                                                        pov=network_info.pov
                                                        ),
                                            num_images=network_info.num_images,
                                            pov=network_info.pov)
                train_angles += inputs[0]
                train_action += inputs[1]
                train_images += inputs[2]

                label_angles += labels[0]
                label_action += labels[1]
                label_target += labels[2]
                label_gripper += labels[3]

                steps += len(inputs[0])
                episode_count += 1
            except (FileNotFoundError, IndexError) as E:
                print(f'[Warn] Received {E}: Reached end of dataset. Using {episode} episodes per '
                      f'network update instead of the {episodes_per_update} usually used')
                break

        h = network.fit(x=[np.asarray(train_angles),
                           np.asarray(train_action),
                           np.asarray(train_images)],
                        y=[np.asarray(label_angles),
                           np.asarray(label_action),
                           np.asarray(label_target),
                           np.asarray(label_gripper)],
                        batch_size=len(train_angles),  # Gradient update after seeing all data in step
                        verbose=0,
                        shuffle=False,
                        epochs=1,  # Epochs are already handled by train_order
                        callbacks=[checkpoint_callback]
                        )

        print()
        print()
        print(h.history)
        print()
        print()

        if virtual_memory().percent > memory_percent_threshold:
            free_memory(memory_percent_threshold)

        if episode_count % display_every == 0 or episode_count == episodes_per_update:
            display_update(network_info=network_info,
                           episode_count=episode_count,
                           total_episodes=total_episodes,
                           start_time=start_time)

    free_memory()

    end_time = time.perf_counter()
    training_time = format_time(end_time - start_time)

    print(f'[Info] Finished training model. Training took {training_time}.')

    ####################
    # Save the network #
    ####################
    network.save(network_save_dir)

    ###################################
    # Update network info and save it #
    ###################################
    save_info_at = join(network_save_dir, 'network_info.pickle')
    with open(save_info_at, 'wb') as handle:
        pickle.dump(network_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ####################
    # Optional testing #
    ####################
    print(network_info.test_amount)
    if network_info.test_amount > 0:
        evaluate_network()

    try:
        plot_model(network, join(network_save_dir, "network.png"), show_shapes=True)
    except ImportError:
        print(f"\n[Warn] Could not print network image. You must install pydot (pip install pydot) "
              f"and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for "
              f"plot_model/model_to_dot to work")

    print(f'\n[Info] Successfully exiting program.')


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
    tf.keras.backend.clear_session()  # Resolves memory overflow
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

# Todo: implement a new evaluation method for multiple steps
#       csv file with new lines added on each call?
def evaluate_network(network, network_save_dir, obs_config, test_info):

    test_dir = test_info['test_dir']
    test_amount = test_info['test_amount']
    test_available = test_info['test_available']

    pov = test_info['pov']

    print(f'\n[info] Beginning to evaluate the model on {test_amount} test demonstration episodes')

    test_order = get_order(test_amount, test_available)

    for episode in test_order:
        inputs, labels = split_data(format_data(load_data(test_dir,
                                                          episode,
                                                          obs_config),
                                                pov=pov),
                                    pov=pov)

        train_angles = inputs[0]
        train_action = inputs[1]
        train_images = inputs[2]

        label_angles = labels[0]
        label_action = labels[1]
        label_target = labels[2]
        label_gripper = labels[3]

        loss = network.evaluate(x=[np.asarray(train_angles),
                                   np.asarray(train_action),
                                   np.asarray(train_images)],
                                y=[np.asarray(label_angles),
                                   np.asarray(label_action),
                                   np.asarray(label_target),
                                   np.asarray(label_gripper)],
                                verbose=1,
                                batch_size=len(train_angles))

    with open(f'{network_save_dir}/model_summary.txt', "w") as f:
        f.write(f'TODO: Implement this!')


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


# Todo: rewrite this function
def save_training_history(train_performance: List, network_save_dir: str, prev_train_performance: np.ndarray = None):
    steps = []
    mse = []
    for perf in train_performance:
        steps.append(perf['steps'])
        mse.append(perf['mse'][0])

    train_performance_csv = np.vstack((np.asarray(steps), np.asarray(mse))).transpose()
    if prev_train_performance is not None:
        train_performance_csv = np.vstack((prev_train_performance,
                                           train_performance_csv))

    np.savetxt(join(network_save_dir, 'train_performance.csv'),
               train_performance_csv,
               delimiter=",",
               header='steps, mse')


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
