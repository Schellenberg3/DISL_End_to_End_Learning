from rlbench.observation_config import ObservationConfig
from tensorflow.keras.models import load_model
from utils.utils import format_data
from utils.utils import load_data
from utils.utils import split_data
from utils.utils import split_data_4
from utils.utils import check_yes
from utils.utils import get_order
from os.path import join
from config import EndToEndConfig
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime

if __name__ == "__main__":
    print('[Info] Starting evaluate.py')

    obs_config = ObservationConfig()

    config = EndToEndConfig()

    network_dir, network_name = config.get_trained_network()
    network = load_model(network_dir)
    print(f'\n[Info] Finished loading the network, {network_name}.')

    parsed_network_name = network_name.split('_')
    if 'pv4' in parsed_network_name or 'rnn-pv4' in parsed_network_name:
        print(f'\n[Info] Detected that the network uses 4 images. '
              f'Will structure inputs accordingly')
        split = split_data_4
    else:
        split = split_data

    pov = config.get_pov_from_name(parsed_network_name)

    eval_dir, eval_amount, eval_available = config.get_evaluate_directory()

    if not check_yes('\nAre you ready to begin? (y/n) '):
        exit(f'\n[Warn] Answer not recognized. Exiting program without creating a new model.')

    eval_order = get_order(eval_amount, eval_available)

    try:
        with open(join(network_dir, 'train_performance.plk'), 'rb') as fp:
            history = pickle.load(fp)

        print(f'\n[Info] Finished loading the network training history. Displaying graph now.')
        steps = []
        mse = []
        for h in history:
            steps.append(h['steps'])
            mse.append(h['mse'][0])

        # todo resolve display issue
        # plt.plot(steps, mse)
        # plt.xlabel('Steps')
        # plt.ylabel('MSE')
        # plt.title('Training MSE')
        # plt.grid(True)
        # plt.show()
        # plt.savefig(join(network_dir, 'training_mse.png'))
    except FileNotFoundError:
        print(f'[Warn] No pickle was found in the {network_name} directory. Skipping graph.')

    print(f'\n[info] Beginning to evaluate the model on {eval_amount} episodes')

    max_mse = -1
    max_mse_ep = -1
    min_mse = float('inf')
    min_mse_ep = -1
    avg_mse = 0
    tot_mse = 0

    for episode in eval_order:
        test_pose, test_view, test_label = split(format_data(load_data(eval_dir,
                                                                       episode,
                                                                       obs_config)),
                                                 pov)

        loss, acc, mse = network.evaluate(x=[np.asarray(test_pose),
                                             np.asarray(test_view)],
                                          y=np.asarray(test_label),
                                          verbose=1,
                                          batch_size=len(test_pose))

        avg_mse += acc

        if mse > max_mse:
            max_mse = mse
            max_mse_ep = episode
        elif mse < min_mse:
            min_mse = mse
            min_mse_ep = episode

    try:
        avg_acc = avg_mse / len(eval_order)
    except ZeroDivisionError:
        avg_acc = 'NO EVALUATION'

    evaluation_summary = f'Evaluated at {datetime.datetime.now()} \n' \
                         f'Tested on {eval_dir} \n' \
                         f'The {len(eval_order)} episodes were numbers {eval_order} \n' \
                         f'Found an average accuracy of {avg_acc}% with a max of {max_mse} ' \
                         f'at episode {max_mse_ep} and a min of {min_mse} at episode {min_mse_ep}.' \
                         f'\n{eval_amount} episodes were used for testing.'

    print('\nEvaluation Summary:')
    print(evaluation_summary)

    with open(join(network_dir, 'model_summary.txt'), "a") as f:
        f.write(f'\n\n-------------------------------------------------------\n\n'
                f'{evaluation_summary}')

    print(f'\n[Info] Successfully exiting program.')
