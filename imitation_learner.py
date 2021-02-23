import numpy as np
from tensorflow import keras
from os import listdir
import os
import time
from rlbench.observation_config import ObservationConfig
from disl_networks import rnn_vision
from disl_networks import rnn_position_vision
from disl_networks import rnn_position_vision_4
from disl_utils import get_order
from disl_utils import load_data
from disl_utils import format_data
from disl_utils import split_data
from disl_utils import split_data_4


if __name__ == "__main__":
    """------ USER VARIABLES -----"""

    # Network options. See disl_networks for explanation of cnn_settings.
    choice = "rnn_position_vision"
    cnn_settings = "Hermann"

    train_path = 'datasets/training/DislPickUpBlueCup/variation0/episodes'
    train_episodes = -1
    epochs = 1

    test_path = 'datasets/training/DislPickUpBlueCup/variation0/episodes'
    test_episodes = 20

    # Settings for compilation. Generally do not need to adjust these.
    use_optimizer = "adam"
    use_loss = "mean_squared_error"
    use_metrics = ["accuracy", "mse"]

    """----- SET UP -----"""

    print(f'[Info] Beginning to compile the {choice} model with cnn_settings: {cnn_settings}')

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    models = {
        "rnn_vision": (rnn_vision, split_data),
        "rnn_position_vision": (rnn_position_vision, split_data),
        "rnn_position_vision_4": (rnn_position_vision_4, split_data_4),
    }
    model = models[choice][0](cnn_settings)
    split = models[choice][1]

    model.compile(optimizer=use_optimizer,
                  loss=use_loss,
                  metrics=use_metrics)

    save_location = f'imitation_trained/{choice}_{cnn_settings}_{train_episodes}_ep'

    try:
        os.listdir(save_location)
        print(f'[WARN] There is already a network at {save_location} training will override this. '
              f' Are you sure you would like to do proceed? (y/n)')
        ans = input()
        if ans not in ['y', 'yes', 'Y', 'Yes']:
            print(f'[Warn] Answer: {ans} not recognized. Exiting program without overriding the exiting model.')
            exit()
    except FileNotFoundError:
        pass

    print(f'\n[Info] Finished compiling the {choice} model with cnn_settings: {cnn_settings}')
    print(f'[Info] Will train with {train_episodes} demonstration episodes over {epochs} epochs.')
    print(f'[Info] Training demonstration episodes will be pulled from: {train_path}')
    print(f'[Info] {test_episodes} testing demonstration episodes will be pulled from: {test_path}')

    try:
        available_training = len(listdir(train_path))
        if available_training < train_episodes:
            print(f'[ERROR] Exiting program. Only {available_training} demonstration episodes are '
                  f'available for training at {train_path}, not the requested {train_episodes}')
            exit()
        elif train_episodes == -1:
            train_episodes = available_training
    except FileNotFoundError:
        pass

    try:
        available_testing = len(listdir(test_path))
        if available_testing < train_episodes:
            print(f'[ERROR] Exiting program. Only {available_testing} demonstration episodes are '
                  f'available for testing at {test_path}, not the requested {test_episodes}')
            exit()
        elif test_episodes == -1:
            test_episodes = available_testing
    except FileNotFoundError:
        print(f'[ERROR] Exiting program. It seems like no files exist at {test_path}')
        exit()

    if True:
        print(f'\n[Info] Ready to begin training on {train_episodes} training demonstration followed by '
              f'testing on {test_episodes} '
              f'Are you ready to begin? (y/n)')
        ans = input()
        if ans not in ['y', 'yes', 'Y', 'Yes']:
            print(f'[Warn] Answer: {ans} not recognized. Exiting program without creating a new model.')
            exit()

        train_order = get_order(train_episodes, epochs)

        start_train = time.perf_counter()

        # todo: write function to view/validate that training data is loaded properly
        # todo: start data from random initial position
        # todo: make sure that vision only is loaded properly ( a function 'get_x_y()'?)
        # todo learn more about array splicing in numpy

        for train in train_order:
            demo = load_data(train_path, train, obs_config)
            demo = format_data(demo)
            train_data, train_images, train_label = split(demo)

            model.fit(x=[np.asarray(train_data),
                         np.asarray(train_images)],
                      y=np.asarray(train_label),
                      verbose=0,
                      shuffle=False,
                      epochs=epochs,
                      workers=os.cpu_count(),
                      use_multiprocessing=True)

        end_train = time.perf_counter()

        print(f'[Info] Finished training model. Training took {(end_train - start_train)} seconds.')

        # todo: verify that the save image and summary work properly
        model.save(save_location)

        keras.utils.plot_model(model, f'{save_location}/model_image.png', show_shapes=True)
        model.summary()

        print(f'[Info] Saved the model at: {save_location}')

    print(f'[info] Beginning to evaluate the model on {test_episodes} test demonstration episodes')

    max_acc = -1
    max_acc_num = -1
    min_acc = float('inf')
    min_acc_num = -1
    avg_acc = 0

    test_order = get_order(test_episodes)
    tot_acc = 0

    for test in test_order:
        print(test)
        demo = load_data(test_path, test, obs_config)
        demo = format_data(demo)
        test_data, test_images, test_label = split_data(demo)

        loss, acc = model.evaluate(x=[np.asarray(test_data),
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

    avg_acc = avg_acc / len(test_order)

    evaluation_summary = f'Found an average accuracy of {avg_acc}% with a max of {max_acc} ' \
                         f'at episode{max_acc_num} and a min of {min_acc} at episode{min_acc_num}'

    print('[Info] ', evaluation_summary)

    with open(f'{save_location}/model_summary.txt', "w") as f:
        f.write(f'Training of {choice} on {train_episodes} took a total time of '
                f'{(end_train - start_train)} seconds.\n\n')
        f.write(evaluation_summary + '\n\n')
        f.write(model.summary())

    print(f'[Info] Successfully exiting program.')
