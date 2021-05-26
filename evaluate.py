from rlbench.observation_config import ObservationConfig

from tensorflow.keras.models import load_model
from tensorflow.keras import Model

from utils.network_info import NetworkInfo
from utils.utils import get_order
from utils.utils import split_data
from utils.utils import format_data
from utils.utils import load_data
from config import EndToEndConfig

from datetime import datetime as dt
from os.path import join

import numpy as np
import pickle


def main():
    print('[Info] Starting evaluate.py')

    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True

    config = EndToEndConfig()
    network_dir = config.get_trained_network()

    with open(join(network_dir, 'network_info.pickle'), 'rb') as f:
        network_info: NetworkInfo = pickle.load(f)

    test_dir, test_amount, test_available = config.get_evaluate_directory()
    network_info.test_dir = test_dir
    network_info.test_amount = test_amount
    network_info.test_available = test_available

    network = load_model(network_dir)

    evaluate_network(network=network,
                     network_info=network_info,
                     network_save_dir=network_dir,
                     obs_config=obs_config)


def evaluate_network(network: Model, network_info: NetworkInfo,
                     network_save_dir: str, obs_config: ObservationConfig) -> None:
    print(f'\n[info] Beginning to evaluate the model on {network_info.test_amount} test demonstration episodes')

    test_order = get_order(network_info.test_amount, network_info.test_available)
    evaluation_performance = []

    for episode in test_order:
        inputs, labels = split_data(format_data(load_data(network_info.test_dir,
                                                          episode,
                                                          obs_config),
                                                pov=network_info.pov),
                                    pov=network_info.pov)

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

        loss.append(episode)
        evaluation_performance.append(loss)

    evaluation_performance = np.array(evaluation_performance)

    fname = 'evaluation'
    task = network_info.test_dir.split('/')[-3]
    time = dt.now().strftime("%Y_%d_%m_%H:%M")

    fname = '_'.join([fname, task, time]) + '.csv'

    header = ['loss', 'joint MSE loss', 'action sparse entropy loss', 'target MSE loss',
              'gripper MSE loss', 'joint RMS', 'action accuracy', 'target RMS', 'gripper RMS', 'episode']
    np.savetxt(join(network_save_dir, fname),
               evaluation_performance,
               delimiter=",",
               header=', '.join(header))


if __name__ == "__main__":
    main()
