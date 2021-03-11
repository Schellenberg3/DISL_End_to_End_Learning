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


    if not check_yes('\nAre you ready to begin? (y/n) '):
        exit(f'\n[Warn] Answer not recognized. Exiting program without creating a new model.')
    print('')

    train_order = get_order(train_amount, epochs)



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
