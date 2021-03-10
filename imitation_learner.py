from rlbench.observation_config import ObservationConfig
from utils import get_order
from utils import load_data
from utils import format_data
from utils import split_data
from contextlib import redirect_stdout
from utils import EndToEndConfig
from utils import check_yes
from utils import format_time
from os.path import join
from psutil import virtual_memory
import numpy as np
import tensorflow as tf
import gc
import pickle
import datetime
import os
import time

if __name__ == "__main__":
    print('[Info] Starting imitation_learner.py')

    """------ USER VARIABLES -----"""

    config = EndToEndConfig()
    train_dir, test_dir = config.set_directories()
    (train_amount, test_amount), epochs = config.get_episode_amounts(train_dir, test_dir)
    network_name, network, split = config.get_new_network()

    # How many episodes should the network see before back propagation
    episodes_per_update = 3

    # Settings for network compilation. Generally do not need to adjust these.
    use_optimizer = "adam"
    use_loss = "mean_squared_error"
    use_metrics = ["accuracy", "mse"]

    """----- SET UP -----"""

    obs_config = ObservationConfig()

    network.compile(optimizer=use_optimizer,
                    loss=use_loss,
                    metrics=use_metrics)

    print(f'\n[Info] Finished compiling the network.')

    network_save_dir = join(config.network_root,
                            'imitation',
                            f'{network_name}_{train_amount}_by_{epochs}')

    print(f'\n[Info] The network will be saved in {network_save_dir}')
    try:
        os.listdir(network_save_dir)
        print(f'\n[WARN] There is already a network at {network_save_dir} training will override this.')
        if not check_yes('Are you sure you would like to do override this? (y/n) '):
            exit(f'\n[Warn] Answer not recognized. Exiting program without overriding the exiting model.')
    except FileNotFoundError:
        pass

    print(f'\n[Info] Pre-training summary: ')
    print(f'[Info] Will train with {train_amount} episodes over {epochs} '
          f'epochs with {test_amount} testing episodes.')
    print(f'[Info] Training episodes will be pulled from: {train_dir}')
    print(f'[Info] Testing episodes will be pulled from: {test_dir}')
    print(f'[Info] The network will be saved at: {network_save_dir}')

    if not check_yes('\nAre you ready to begin? (y/n) '):
        exit(f'\n[Warn] Answer not recognized. Exiting program without creating a new model.')
    print('')

    train_order = get_order(train_amount, epochs)

    start_train = time.perf_counter()

    # todo refactor names
    # Todo solve memory creep
    step = 0
    total_steps = len(train_order) - 1
    display_every = int(np.ceil(len(train_order)/100))

    train_performance = []
    time_steps = []
    i = 0

    while step <= total_steps:
        train_pose = []
        train_view = []
        train_label = []
        for episode in range(episodes_per_update):
            try:
                pose, view, label = split(format_data(load_data(train_dir,
                                                                train_order[step],
                                                                obs_config)))
                train_pose += pose
                train_view += view
                train_label += label
                i += len(train_pose)
                # print(f'Loaded ep {train_order[step]} at step {step} of {total_steps}')
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
            h.history['steps'] = i
            train_performance.append(h.history)
            print(f'[Info] {step/(total_steps + 1) * 100:3.1f}% Complete '
                  f'{format_time((time.perf_counter() - start_train)*(total_steps - step + 1)/(step))} remaining. '
                  f'Trained through episode {step - train_amount*int((step - 1)/train_amount)} of '
                  f'{train_amount} in epoch {int((step - 1)/train_amount)+1} of {epochs}.\n')

    end_train = time.perf_counter()

    print(f'[Info] Finished training model. Training took {format_time(end_train - start_train)} seconds.')

    network.save(network_save_dir)

    print(f'\n[Info] Saved the model at: {network_save_dir}')

    with open(f'{network_save_dir}/model_summary.txt', "w") as f:
        with redirect_stdout(f):
            network.summary()

    with open(f'{network_save_dir}/train_performance.plk', 'wb') as file:
        pickle.dump(train_performance, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'\n[info] Beginning to evaluate the model on {test_amount} test demonstration episodes')

    max_acc = -1
    max_acc_ep = -1
    min_acc = float('inf')
    min_acc_ep = -1
    avg_acc = 0

    test_order = get_order(test_amount)
    tot_acc = 0

    for episode in test_order:
        test_pose, test_view, test_label = split_data(format_data(load_data(test_dir,
                                                                            episode,
                                                                            obs_config)))

        loss, acc, mse = network.evaluate(x=[np.asarray(test_pose), np.asarray(test_view)],
                                          y=np.asarray(test_label),
                                          verbose=1,
                                          batch_size=len(test_pose))

        avg_acc += acc

        if acc > max_acc:
            max_acc = acc
            max_acc_ep = episode
        elif acc < min_acc:
            min_acc = acc
            min_acc_ep = episode

    try:
        avg_acc = avg_acc / len(test_order)
    except ZeroDivisionError:
        avg_acc = 'NO EVALUATION'

    evaluation_summary = f'Evaluated at {datetime.datetime.now()} \n' \
                         f'Found an average accuracy of {avg_acc}% with a max of {max_acc} ' \
                         f'at episode {max_acc_ep} and a min of {min_acc} at episode {min_acc_ep}.' \
                         f'\n{test_amount} episodes were used for testing.'

    print('\nEvaluation Summary:')
    print(evaluation_summary)

    with open(f'{network_save_dir}/model_summary.txt', "w") as f:
        f.write(f'\n\nTraining took a total of {format_time(end_train - start_train)}.'
                f'\n\n{evaluation_summary}')

    print(f'\n[Info] Successfully exiting program.')
