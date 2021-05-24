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

    episode_info = config.get_episode_amounts(train_dir, test_dir)

    task_name, _ = config.get_task_from_name(train_dir.split('/'))
    print()  # to maintain spacing todo: FIX SPACING

    pov = config.get_pov_from_user()

    network, network_name, network_info = config.get_new_network()

    train(network=network,
          network_info=network_info,
          train_dir=train_dir,
          test_dir=test_dir,
          episode_info=episode_info,
          pov=pov,
          network_name=network_name,
          task_name=task_name,
          save_root=config.network_root)


def train_existing(config: EndToEndConfig) -> None:
    print(f'\n[Info] Continuing the training of an existing model')

    network_dir, network_name = config.get_trained_network()
    parsed_name = network_name.split('_')

    network_info = config.get_info_from_network_name(parsed_name)

    pov = network_info[0]
    split = network_info[1]
    task_name = network_info[2]
    task = network_info[3]
    train_dir = network_info[4]
    test_dir = network_info[5]
    train_amount = network_info[6]
    train_available = network_info[7]
    test_amount = network_info[8]
    test_available = network_info[9]
    prev_epoch = network_info[10]

    episode_info = network_info[6:10]

    epochs = int(input(f'\nHow many more epoch (at {prev_epoch} currently)'
                       f' should the network be trained on (default is 1)? ')) or 1
    epochs = 1 if epochs < 1 else epochs
    print(f'Training the network for {epochs} additional epoch. ')

    network = load_model(network_dir)

    pickle_location = join(network_dir, 'network_info.pickle')
    with open(pickle_location, 'rb') as handle:
        network_info = pickle.load(handle)

    # todo write method to save and load this info
    prev_train_performance = None
    prev_max_step = 0

    train(network=network,
          network_info=network_info,
          train_dir=train_dir,
          test_dir=test_dir,
          episode_info=episode_info,
          pov=pov,
          network_name=network_name,
          task_name=task_name,
          save_root=config.network_root,
          prev_train_performance=prev_train_performance,
          prev_epoch=prev_epoch,
          prev_max_step=prev_max_step)


def train(network: Model,
          network_info: Dict,
          train_dir: str,
          test_dir: str,
          episode_info: Tuple[int, int, int, int, int],
          pov: str,
          network_name: str,
          task_name: str,
          save_root: str,
          prev_train_performance: np.ndarray = None,
          prev_epoch: int = 0,
          prev_max_step: int = 0,
          ):

    # todo: better labels for what each of these pieces of info before the training loop is for
    train_amount = episode_info[0]
    train_available = episode_info[1]

    test_amount = episode_info[2]
    test_available = episode_info[3]

    epochs = episode_info[4]

    num_images = network_info['num_images']

    save_network_as = f'{network_name}_{task_name}_{pov}_{train_amount}_by{epochs + prev_epoch}'
    network_save_dir = join(save_root,
                            'imitation',
                            save_network_as)

    check_if_network_exists(network_save_dir)

    print(f'\n[Info] Pre-training summary: ')
    print(f'Will train with {train_amount} episodes over {epochs} '
          f'Training episodes will be pulled from: {train_dir}\n'
          f'The network will be saved at: {network_save_dir}')

    input('\nReady to begin training. Press enter to proceed...')

    train_order = get_order(train_amount, train_available, epochs)

    start_train = time.perf_counter()

    step = 0
    total_steps = len(train_order) - 1
    display_every = int(np.ceil(len(train_order) / 100))

    train_performance = []
    i = 0

    # How many episodes should the network see before back propagation
    episodes_per_update = 2

    checkpoint_callback = ModelCheckpoint(filepath='',
                                          save_weights_only=False,
                                          save_freq=100_000_000)

    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True

    memory_percent_threshold = 70

    while step <= total_steps:
        train_angles = []
        train_action = []
        train_images = []

        label_angles = []
        label_action = []
        label_target = []
        label_gripper = []

        for episode in range(episodes_per_update):
            try:
                inputs, labels = split_data(format_data(load_data(train_dir,
                                                                  train_order[step],
                                                                  obs_config),
                                                        pov=pov
                                                        ),
                                            num_images=num_images,
                                            pov=pov)
                train_angles += inputs[0]
                train_action += inputs[1]
                train_images += inputs[2]

                label_angles += labels[0]
                label_action += labels[1]
                label_target += labels[2]
                label_gripper += labels[3]

                i += len(inputs[0])
                step += 1
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

        if step % display_every == 0 or step == episodes_per_update:
            h.history['steps'] = i + prev_max_step
            train_performance.append(h.history)
            print(f'[Info] {step / (total_steps + 1) * 100:3.1f}% Complete '
                  f'{format_time((time.perf_counter() - start_train) * (total_steps - step + 1) / (step))} remaining. '
                  f'Trained through episode {step - train_amount * int((step - 1) / train_amount)} of '
                  f'{train_amount} in epoch {int((step - 1) / train_amount) + 1 + prev_epoch} of'
                  f' {epochs + prev_epoch}.\n')

    free_memory()
    end_train = time.perf_counter()

    training_time = format_time(end_train - start_train)

    print(f'[Info] Finished training model. Training took {training_time}.')

    network.save(network_save_dir)

    if not prev_train_performance:  # If it is not an existing network we want to save its info now
        save_info_at = join(network_save_dir, 'network_info.pickle')
        with open(save_info_at, 'wb') as handle:
            pickle.dump(network_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if test_amount > 0:
        evaluate_network(network=network,
                         network_save_dir=network_save_dir,
                         obs_config=obs_config,
                         pov=pov,
                         test_dir=test_dir,
                         test_amount=test_amount,
                         test_available=test_available)

    try:
        plot_model(network, join(network_save_dir, "network.png"), show_shapes=True)
    except ImportError:
        print(f"\n[Warn] Could not print network image. You must install pydot (pip install pydot) and "
              f"install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model/model_to_dot to work")

    print(f'\n[Info] Successfully exiting program.')


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


# Todo: implement a new evaluation method for multiple steps
#       csv file with new lines added on each call?
def evaluate_network(network, network_save_dir, obs_config, pov, test_dir, test_amount, test_available):
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
