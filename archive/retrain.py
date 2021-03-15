# This file will load an existing network, detect its POV, it's Task, and select the dataset


from os.path import isdir
from os import listdir
from rlbench.observation_config import ObservationConfig
from utils.utils import get_order
from utils.utils import load_data
from utils.utils import format_data
from utils.utils import check_yes
from utils.utils import format_time
from utils.utils import split_data
from utils.utils import split_data_4
from tensorflow.keras.models import load_model
from os.path import join
from config import EndToEndConfig
from psutil import virtual_memory
import numpy as np
import tensorflow as tf
import gc
import time

if __name__ == '__main__':
    print('[info] Starting retrain.py')

    config = EndToEndConfig()

    if check_yes('\nContinue training an existing model (y) or train a new model (n): '):

        retrain = True

        network_name, network_dir = config.get_trained_network()

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

        try:
            epochs = int(input(f'\nHow many more epoch (at {prev_epoch} currently)'
                               f' should the network be trained on? '))
            if epochs < 1:
                print('[Warn] Epochs must be greater than 1. Will train for just 1 epoch.')
                epochs = 1
        except TypeError:
            print('[Warn] Epochs must be an integer greater than 1. Will train for just 1 epoch.')
            epochs = 1

        network = load_model(join(config.network_root,
                                  'imitation',
                                  network_dir))

        final_epochs = 'by' + str(prev_epoch + epochs)

        i = -2 if 'rand' in parsed_name else -1
        parsed_name[i] = final_epochs
        network_name = '_'.join(parsed_name)

        prev_train_performance = np.loadtxt(join(config.network_root,
                                                 'imitation',
                                                 network_dir,
                                                 'train_performance.csv'),
                                            delimiter=",")

        pre_max_step = prev_train_performance[-1, 0]

    else:
        prev_train_performance = None
        prev_epoch = 0
        pre_max_step = 0
        exit('Would create a new network')


    network_save_dir = join(config.network_root,
                            'imitation',
                            network_name)

    print(f'\n[Info] The network will be saved in {network_save_dir}')
    try:
        listdir(network_save_dir)
        print(f'\n[WARN] There is already a network at {network_save_dir} training will override this.')
        if not check_yes('Are you sure you would like to do override this? (y/n) '):
            exit(f'\n[Warn] Answer not recognized. Exiting program without overriding the exiting model.')
    except FileNotFoundError:
        pass

    print(f'\n[Info] Pre-training summary: ')
    print(f'Will train with {train_amount} episodes over {epochs} '
          f'epochs with {test_amount} testing episodes. \n'
          f'Training episodes will be pulled from: {train_dir}\n'
          f'Testing episodes will be pulled from: {test_dir}\n'
          f'The network will be saved at: {network_save_dir}')

    if not check_yes('\nAre you ready to begin? (y/n) '):
        exit(f'\n[Warn] Answer not recognized. Exiting program without creating a new model.')
    print('')  # to maintain spacing
    train_order = get_order(train_amount, train_available, epochs)

    start_train = time.perf_counter()

    step = 0
    total_steps = len(train_order) - 1
    display_every = int(np.ceil(len(train_order) / 100))

    train_performance = []
    time_steps = []
    i = 0

    # How many episodes should the network see before back propagation
    episodes_per_update = 3

    obs_config = ObservationConfig()

    while step <= total_steps:
        train_pose = []
        train_view = []
        train_label = []
        for episode in range(episodes_per_update):
            try:
                pose, view, label = split(format_data(load_data(train_dir,
                                                                train_order[step],
                                                                obs_config)),
                                          pov=pov)
                train_pose += pose
                train_view += view
                train_label += label
                i += len(train_pose)
                step += 1
            except (FileNotFoundError, IndexError) as E:
                print(f'[Warn] Reached end of dataset. Using {episode} '
                      f'episodes per network update instead of the {episodes_per_update} '
                      f'usually used')
                break

        h = network.fit(x=[np.asarray(train_pose),
                           np.asarray(train_view)],
                        y=np.asarray(train_label),
                        batch_size=len(train_pose),  # Gradient update after seeing all data in step
                        verbose=0,
                        shuffle=False,
                        epochs=1,  # Epochs are already handled by train_order
                        )

        if virtual_memory().percent > 80:
            # Prevents memory overflow in custom training loops
            #     see: https://github.com/tensorflow/tensorflow/issues/37505
            tf.keras.backend.clear_session()  # Resolves memory overflow
            gc.collect()  # Explicitly called for safety. Does not affect speed.
            print(f'[Warn] Over 80% memory in use. Called functions '
                  f'to clear free unused memory. Training will proceed.')

        if step % display_every == 0 or step == episodes_per_update:
            h.history['steps'] = i + pre_max_step
            train_performance.append(h.history)
            print(f'[Info] {step / (total_steps + 1) * 100:3.1f}% Complete '
                  f'{format_time((time.perf_counter() - start_train) * (total_steps - step + 1) / (step))} remaining. '
                  f'Trained through episode {step - train_amount * int((step - 1) / train_amount)} of '
                  f'{train_amount} in epoch {int((step - 1) / train_amount) + 1 + prev_epoch} of'
                  f' {epochs + prev_epoch}.\n')

    end_train = time.perf_counter()

    training_time = format_time(end_train - start_train)

    print(f'[Info] Finished training model. Training took {training_time}.')

    network.save(network_save_dir)

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


