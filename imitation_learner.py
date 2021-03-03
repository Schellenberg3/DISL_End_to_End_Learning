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
import numpy as np
import os
import time

if __name__ == "__main__":
    # todo add functionality to load a saved model and evaluate it
    print('[Info] Starting imitation_learner.py')
    """------ USER VARIABLES -----"""

    config = EndToEndConfig()
    train_dir, test_dir = config.set_directories()
    (train_amount, test_amount), epochs = config.get_episode_amounts(train_dir, test_dir)
    network_name, network, split = config.get_network()

    # Settings for network compilation. Generally do not need to adjust these.
    use_optimizer = "adam"
    use_loss = "mean_squared_error"
    use_metrics = ["accuracy", "mse"]

    """----- SET UP -----"""

    obs_config = ObservationConfig()
    obs_config.set_all(True)

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

    # todo: write function to view/validate that training data is loaded properly
    # todo: start data from random initial position
    # todo: make sure that vision only is loaded properly ( a function 'get_x_y()'?)
    # todo learn more about array splicing in numpy

    total_steps = epochs*train_amount
    display_every = int(total_steps/100) + 1
    for i, train in enumerate(train_order):
        demo = load_data(train_dir, train, obs_config)
        demo = format_data(demo)
        train_data, train_images, train_label = split(demo)

        network.fit(x=[np.asarray(train_data),
                       np.asarray(train_images)],
                    y=np.asarray(train_label),
                    verbose=0,
                    shuffle=False,
                    epochs=epochs,
                    workers=os.cpu_count(),
                    use_multiprocessing=True)

        if i % display_every == 0:
            print(f'[Info] {i/total_steps * 100:3.1f}% Complete '
                  f'{format_time((time.perf_counter() - start_train)*(total_steps - i)/(i+1))} remaining. '
                  f'At episode {i - train_amount*int(i/train_amount) + 1} of {train_amount} '
                  f'in epoch {int(i/train_amount)+1} of {epochs}.')

    end_train = time.perf_counter()

    print(f'\n[Info] Finished training model. Training took {format_time(end_train - start_train)} seconds.')

    # todo: verify that the save image and summary work properly
    network.save(network_save_dir)

    print(f'\n[Info] Saved the model at: {network_save_dir}')

    print(f'\n[info] Beginning to evaluate the model on {test_amount} test demonstration episodes')

    max_acc = -1
    max_acc_num = -1
    min_acc = float('inf')
    min_acc_num = -1
    avg_acc = 0

    test_order = get_order(test_amount)
    tot_acc = 0

    for test in test_order:
        print(test)
        demo = load_data(test_data, test, obs_config)
        demo = format_data(demo)
        test_data, test_images, test_label = split_data(demo)

        loss, acc = network.evaluate(x=[np.asarray(test_data),
                                        np.asarray(test_images)],
                                     y=[np.asarray(test_label)],
                                     verbose=0,
                                     workers=os.cpu_count(),
                                     use_multiprocessing=True)

        print(acc)
        print(type(acc))

        avg_acc += acc

        if acc > max_acc:
            max_acc = acc
            max_acc_num = test
        elif acc < min_acc:
            min_acc = acc
            min_acc_num = test

    try:
        avg_acc = avg_acc / len(test_order)
    except ZeroDivisionError:
        avg_acc = 'NO EVALUATION'

    evaluation_summary = f'Found an average accuracy of {avg_acc}% with a max of {max_acc} ' \
                         f'at episode{max_acc_num} and a min of {min_acc} at episode{min_acc_num}.' \
                         f'\n\n{test_amount} episodes were used for testing.'

    print('[Info] ', evaluation_summary)

    with open(f'{network_save_dir}/model_summary.txt', "w") as f:
        with redirect_stdout(f):
            network.summary()
        f.write(f'\n\nTraining took a total of {format_time(end_train - start_train)}.'
                f'\n\n{evaluation_summary}')

    print(f'[Info] Successfully exiting program.')
