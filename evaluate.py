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
    eval_dir, eval_amount = config.get_evaluate_directory()

    network_dir, network_name = config.get_trained_network()
    network = load_model(network_dir)

    if 'pv4' in network_name.split('_') or 'rnn-pv4' in network_name.split('_'):
        split = split_data_4
    else:
        split = split_data

    print(f'\n[Info] Finished loading the network, {network_name}.')

    if not check_yes('\nAre you ready to begin? (y/n) '):
        exit(f'\n[Warn] Answer not recognized. Exiting program without creating a new model.')

    eval_order = get_order(eval_amount)

    with open(join(network_dir, 'train_performance.plk'), 'rb') as fp:
        history = pickle.load(fp)

    print(f'\n[Info] Finished loading the network training history. Displaying graph now.')
    steps = []
    mse = []
    for i in history:
        steps.append(history[i]['steps'])
        mse.append(history[i]['mse'])

    plt.plot(steps, mse)
    plt.xlabel('Steps')
    plt.ylabel('MSE')
    plt.title('Training MSE')
    plt.grid(True)
    plt.show()
    plt.savefig(join(network_dir, 'training_mse.png'))

    print(f'\n[info] Beginning to evaluate the model on {eval_amount} episodes')

    max_acc = -1
    max_acc_ep = -1
    min_acc = float('inf')
    min_acc_ep = -1
    avg_acc = 0
    tot_acc = 0

    for episode in eval_order:
        test_pose, test_view, test_label = split(format_data(load_data(eval_dir,
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
        avg_acc = avg_acc / len(eval_order)
    except ZeroDivisionError:
        avg_acc = 'NO EVALUATION'

    evaluation_summary = f'Evaluated at {datetime.datetime.now()} \n' \
                         f'Found an average accuracy of {avg_acc}% with a max of {max_acc} ' \
                         f'at episode {max_acc_ep} and a min of {min_acc} at episode {min_acc_ep}.' \
                         f'\n{eval_amount} episodes were used for testing.'

    print('\nEvaluation Summary:')
    print(evaluation_summary)

    with open(join(network_dir, 'model_summary.txt'), "w") as f:
        f.write(f'\n\n-------------------------------------------------------\n\n'
                f'{evaluation_summary}')

    print(f'\n[Info] Successfully exiting program.')
