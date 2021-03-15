from rlbench.observation_config import ObservationConfig
from contextlib import redirect_stdout
from utils.utils import get_order
from utils.utils import load_data
from utils.utils import format_data
from utils.utils import check_yes
from utils.utils import format_time
from os.path import join
from config import EndToEndConfig
from psutil import virtual_memory
import numpy as np
import tensorflow as tf
import gc
import datetime
import os
import time

if __name__ == "__main__":
    print('[Info] Starting imitation_learner.py')

    config = EndToEndConfig()

    train_dir, test_dir = config.get_train_test_directories()
    episode_info = config.get_episode_amounts(train_dir, test_dir)

    train_amount, train_available, test_amount, test_available, epochs = episode_info

    task_name, _ = config.get_task_from_name(train_dir.split('/'))
    print()  # to maintain spacing

    pov = config.get_pov_from_user()

    network_name, network, split = config.get_new_network()

    # append more information to get the final network name
    network_name = f'{network_name}_{task_name}_{pov}_{train_amount}_by{epochs}'

    # Settings for network compilation. Generally do not need to adjust these.
    use_optimizer = "adam"
    use_loss = "mean_squared_error"
    use_metrics = ["mse"]

    network.compile(optimizer=use_optimizer,
                    loss=use_loss,
                    metrics=use_metrics)

    print(f'\n[Info] Finished compiling the network.')

    network_save_dir = join(config.network_root,
                            'imitation',
                            f'{network_name}')

    print(f'\n[Info] The network will be saved in {network_save_dir}')
    try:
        os.listdir(network_save_dir)
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
    display_every = int(np.ceil(len(train_order)/100))

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
            h.history['steps'] = i
            train_performance.append(h.history)
            print(f'[Info] {step/(total_steps + 1) * 100:3.1f}% Complete '
                  f'{format_time((time.perf_counter() - start_train)*(total_steps - step + 1)/(step))} remaining. '
                  f'Trained through episode {step - train_amount*int((step - 1)/train_amount)} of '
                  f'{train_amount} in epoch {int((step - 1)/train_amount)+1} of {epochs}.\n')

    end_train = time.perf_counter()

    training_time = format_time(end_train - start_train)

    print(f'[Info] Finished training model. Training took {training_time}.')

    network.save(network_save_dir)

    print(f'\n[Info] Saved the model at: {network_save_dir}')

    with open(f'{network_save_dir}/model_summary.txt', "w") as f:
        with redirect_stdout(f):
            network.summary()

    steps = []
    mse = []
    for perf in train_performance:
        steps.append(perf['steps'])
        mse.append(perf['mse'][0])

    train_performance_csv = np.vstack((np.asarray(steps), np.asarray(mse))).transpose()

    np.savetxt(join(network_save_dir, 'train_performance.csv'),
               train_performance_csv,
               delimiter=",",
               header='steps, mse')

    print(f'\n[info] Beginning to evaluate the model on {test_amount} test demonstration episodes')

    test_order = get_order(test_amount, test_available)

    max_mse = -1
    max_mse_ep = -1
    min_mse = float('inf')
    min_mse_ep = -1
    avg_mse = 0
    tot_mse = 0

    for episode in test_order:
        test_pose, test_view, test_label = split(format_data(load_data(test_dir,
                                                                       episode,
                                                                       obs_config)),
                                                 pov)

        loss, mse = network.evaluate(x=[np.asarray(test_pose),
                                        np.asarray(test_view)],
                                     y=np.asarray(test_label),
                                     verbose=1,
                                     batch_size=len(test_pose))

        avg_mse += mse

        if mse > max_mse:
            max_mse = mse
            max_mse_ep = episode
        elif mse < min_mse:
            min_mse = mse
            min_mse_ep = episode

    try:
        avg_acc = avg_mse / len(test_order)
    except ZeroDivisionError:
        avg_acc = 'NO EVALUATION'

    evaluation_summary = f'Network created and evaluated at {datetime.datetime.now()} \n' \
                         f'Training directory was: {train_dir}\n' \
                         f'Training took {training_time} \n' \
                         f'Testing directory was: {test_dir} \n' \
                         f'Number of testing episodes was {test_amount} \n' \
                         f'Found an average mse of {avg_acc} with a max of {max_mse} ' \
                         f'at episode {max_mse_ep} and a min of {min_mse} at episode {min_mse_ep}.'

    print('\nEvaluation Summary:')
    print(evaluation_summary)

    with open(f'{network_save_dir}/model_summary.txt', "w") as f:
        f.write(f'\n\nTraining took a total of {format_time(end_train - start_train)}.'
                f'\n\n{evaluation_summary}')

    print(f'\n[Info] Successfully exiting program.')
