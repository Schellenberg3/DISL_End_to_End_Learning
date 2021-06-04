from rlbench.observation_config import ObservationConfig

from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

from utils.network_info import NetworkInfo
from utils.utils import format_data
from utils.utils import split_data
from utils.utils import load_data
from utils.utils import get_order

from config import EndToEndConfig

from datetime import datetime as dt
from random import uniform

from os.path import join
from os import listdir

from argparse import Namespace
import argparse

import numpy as np
import pickle
import cv2


def main(arguments: Namespace):
    print('[Info] Starting evaluate.py')

    visual = arguments.vis

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

    network = load_model(join(network_dir, network_dir.split('/')[-1] + '.h5'))

    if visual:
        display_vis(network=network,
                    network_info=network_info,
                    obs_config=obs_config)
    else:
        evaluate_network(network=network,
                         network_info=network_info,
                         network_save_dir=network_dir,
                         obs_config=obs_config)


def display_vis(network, network_info, obs_config):

    num_available = len(listdir(network_info.test_dir))
    ep_num = int(input(f'\nPlease input the episode number to view '
                       f'(episodes 0 to {num_available - 1} available. Default random): ') or -1)
    ep_num = int(uniform(0, num_available)) if ep_num < 0 else ep_num

    dir_name = '/'.join(network_info.test_dir.split('/')[-4:-2])

    inputs, labels = split_data(format_data(load_data(network_info.test_dir,
                                                      ep_num,
                                                      obs_config),
                                            pov=network_info.pov
                                            ),
                                num_images=network_info.num_images,
                                pov=network_info.pov)

    test_angles = inputs[0]
    test_action = inputs[1]
    test_images = inputs[2]

    label_angles = labels[0]
    label_action = labels[1]
    label_target = labels[2]
    label_gripper = labels[3]

    width = 128 * network_info.num_images if 128 * network_info.num_images > 1024 else 1024
    height = 800
    blank = np.ones((height, width, 3)) * 0.1

    width_offset = 10
    text_offset = 30

    row_1 = 10
    row_2 = row_1 + 128 + text_offset
    row_3 = row_2 + text_offset
    row_4 = row_3 + 128 + text_offset
    row_5 = row_4 + text_offset
    row_6 = row_5 + text_offset
    row_7 = row_6 + int(1.5*text_offset)
    row_8 = row_7 + text_offset
    row_9 = row_8 + int(1.5*text_offset)
    row_10 = row_9 + text_offset
    row_11 = row_10 + int(1.5*text_offset)
    row_12 = row_11 + text_offset
    row_13 = row_12 + int(1.5*text_offset)
    row_14 = row_13 + text_offset

    font = cv2.FONT_HERSHEY_SIMPLEX

    rms = RootMeanSquaredError()
    sca = SparseCategoricalAccuracy()
    sca.reset_states()

    step = 0
    total_steps = len(test_angles)

    while step in range(total_steps):
        display = blank.copy()

        #########################
        # Top row of RGB Images #
        #########################
        for i in range(network_info.num_images):
            tmp_rgb = test_images[step][:, :, 4 * i:4 * i + 3]
            display[row_1:row_1 + 128, 128 * i:128 * i + 128, :] = tmp_rgb

        ####################################
        # Row of text with image max / min #
        ####################################
        rgb_text = f'RGB values: max={np.max(test_images[step][:, :, 0:4]):.3f} ' \
                   f'min={np.min(test_images[step][:, :, 0:4]):.3f}'
        cv2.putText(img=display,
                    text=rgb_text,
                    org=(width_offset, row_2),
                    fontFace=font,
                    fontScale=0.75,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        #######################
        # Row of depth images #
        #######################
        for i in range(network_info.num_images):
            tmp_depth = test_images[step][:, :, i*4 + 3]
            scaled_temp_depth = tmp_depth / np.max(tmp_depth)
            display[row_3:row_3 + 128, 128 * i:128 * i + 128, 0] = scaled_temp_depth

        ###############################
        # Row of text with depth info #
        ###############################
        rgb_text = f'Depth values: max={np.max(test_images[step][:, :, 3]):.3f} ' \
                   f'min={np.min(test_images[step][:, :, 3]):.3f}'
        cv2.putText(img=display,
                    text=rgb_text,
                    org=(width_offset, row_4),
                    fontFace=font,
                    fontScale=0.75,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ##############################
        # Row with the gripper input #
        ##############################
        input_action_text = f'Input gripper action: {test_action[step]} (0 := closed, 1 := open))'
        cv2.putText(img=display,
                    text=input_action_text,
                    org=(width_offset, row_5),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        #########################################
        # Row with the scaled joint angle input #
        #########################################
        input_angle_text = ''
        input_angle_text = '[' + ', '.join([input_angle_text + f'{p:.3f}' for p in test_angles[step]]) + ']'
        input_angle_text = f'Input joint angles: {input_angle_text})'
        cv2.putText(img=display,
                    text=input_angle_text,
                    org=(width_offset, row_6),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ######################
        # Predict the values #
        ######################
        prediction = network.predict(x=[np.expand_dims(test_angles[step], 0),
                                        np.expand_dims(test_action[step], 0),
                                        np.expand_dims(test_images[step], 0)])
        out_angles = prediction[0].flatten()
        out_action = prediction[1].flatten()
        out_target = prediction[2].flatten()
        out_gripper = prediction[3].flatten()

        ########################
        # Two rows to display: #
        # - label action       #
        # - predicted action   #
        ########################
        angle_mse = rms(label_angles[step], out_angles).numpy()
        label_angle_text = ''
        label_angle_text = '[' + ', '.join([label_angle_text + f'{p:.3f}' for p in label_angles[step]]) + ']'
        label_angle_text = f'Input joint angles:  {label_angle_text}... RMS {angle_mse:.4f}'
        cv2.putText(img=display,
                    text=label_angle_text,
                    org=(width_offset, row_7),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        out_angle_text = ''
        out_angle_text = '[' + ', '.join([out_angle_text + f'{p:.3f}' for p in out_angles]) + ']'
        out_angle_text = f'Output joint angles: {out_angle_text}'
        cv2.putText(img=display,
                    text=out_angle_text,
                    org=(width_offset, row_8),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ########################
        # Two rows to display: #
        # - label action       #
        # - predicted action   #
        ########################
        sca.update_state(label_action[step], out_action)
        label_action_text = f'Label gripper action:  {label_action[step]} ' \
                            f'(0 := closed, 1 := open)... ACC {sca.result():.4f}'
        cv2.putText(img=display,
                    text=label_action_text,
                    org=(width_offset, row_9),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        out_action_text = f'Output gripper action: {out_action} -> {np.argmax(out_action)}'
        cv2.putText(img=display,
                    text=out_action_text,
                    org=(width_offset, row_10),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ########################
        # Two rows to display: #
        # - label target       #
        # - predicted target   #
        ########################
        target_mse = rms(label_target[step], out_target).numpy()
        label_target_text = ''
        label_target_text = '[' + ', '.join([label_target_text + f'{p:.3f}' for p in label_target[step]]) + ']'
        label_target_text = f'Label target position:  {label_target_text}... RMS {target_mse:.4f}'
        cv2.putText(img=display,
                    text=label_target_text,
                    org=(width_offset, row_11),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        out_target_text = ''
        out_target_text = '[' + ', '.join([out_target_text + f'{p:.3f}' for p in out_target]) + ']'
        out_target_text = f'Output target position: {out_target_text}'
        cv2.putText(img=display,
                    text=out_target_text,
                    org=(width_offset, row_12),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ########################
        # Two rows to display: #
        # - label gripper      #
        # - predicted gripper  #
        ########################
        gripper_mse = rms(label_gripper[step], out_gripper).numpy()
        label_gripper_text = ''
        label_gripper_text = '[' + ', '.join([label_gripper_text + f'{p:.3f}' for p in label_gripper[step]]) + ']'
        label_gripper_text = f'Label gripper position:  {label_gripper_text}... RMS {gripper_mse:.4f}'
        cv2.putText(img=display,
                    text=label_gripper_text,
                    org=(width_offset, row_13),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        out_gripper_text = ''
        out_gripper_text = '[' + ', '.join([out_gripper_text + f'{p:.3f}' for p in out_gripper]) + ']'
        out_gripper_text = f'Output gripper position: {out_gripper_text}'
        cv2.putText(img=display,
                    text=out_gripper_text,
                    org=(width_offset, row_14),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ###################
        # Bottom of image #
        ###################
        display_info = f'{network_info.network_name}. Demo {ep_num} from {dir_name}...' \
                       f' On step {step} of {total_steps - 1}.'
        cv2.putText(img=display,
                    text=display_info,
                    org=(width_offset, height - text_offset),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        display = display[:, :, [2, 1, 0]]  # OpenCV uses BGR, images saved as RGB. Must convert.
        cv2.imshow('Display', display)
        cv2.waitKey(1)

        cmd = input('Next command (Next := n, previous := p, quit := q. Default n): ') or 'n'

        if cmd == 'n':
            step += 1
        if cmd == 'p':
            step -= 1
            step = 0 if step < 0 else step
        if cmd == 'q':
            break

    cv2.destroyAllWindows()


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
                                    num_images=network_info.num_images,
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
    if 'randomized' in network_info.test_dir.split('/')[-4].split('_'):
        task += '_Rand'
    time = dt.now().strftime("%Y_%d_%m_%H:%M")

    fname = '_'.join([fname, task, time]) + '.csv'

    header = ['loss', 'joint MSE loss', 'action sparse entropy loss', 'target MSE loss',
              'gripper MSE loss', 'joint RMS', 'action accuracy', 'target RMS', 'gripper RMS', 'episode']
    np.savetxt(join(network_save_dir, fname),
               evaluation_performance,
               delimiter=",",
               header=', '.join(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accepts input velocity of the ball.')
    parser.add_argument("--vis", default=False, type=bool)
    args = parser.parse_args()
    main(args)
