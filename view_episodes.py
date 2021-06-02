from rlbench.observation_config import ObservationConfig

from config import EndToEndConfig

from utils.utils import format_data
from utils.utils import split_data
from utils.utils import load_data

from random import uniform

from copy import deepcopy

from os.path import join
from os import listdir

import numpy as np

import cv2


def main():
    config = EndToEndConfig()
    config.list_data_set_directories()

    #######################################
    # Gather users inputs for the display #
    #######################################
    try:
        dir_num = int(input('\nSelect a directory # to check (default 0): ') or 0)
        dataset_dir = join(config.data_root, config.possible_data_set[dir_num], 'variation0', 'episodes')
    except (IndexError, ValueError) as e:
        dir_num = None
        dataset_dir = None
        exit(f'\n[Error] {e}: Selection should be an integer above 0. Exiting program.')

    num_available = len(listdir(dataset_dir))
    ep_num = int(input(f'\nPlease input the episode number to view '
                       f'(episodes 0 to {num_available - 1} available. Default random): ') or -1)
    ep_num = int(uniform(0, num_available)) if ep_num < 0 else ep_num

    pov = input('\nPlease enter the point of view (front or wrist. Default front): ') or 'front'
    num_images = int(input('\nPlease enter the number of images to use as input (Recommended max is 8. '
                           'Default 4): ') or 4)

    ################################################################
    # Load from memory the selected demo and create 3 objects for: #
    #  - the data straight from disk                               #
    #  - the data after being formatted                            #
    #  - the data after its been split                             #
    ################################################################
    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True
    on_disk = load_data(path=dataset_dir,
                        example_num=ep_num,
                        obs_config=obs_config)

    formatted = format_data(episode=deepcopy(on_disk),
                            pov=pov)

    inputs, labels = split_data(episode=deepcopy(formatted),
                                num_images=num_images,
                                pov=pov)

    train_angles = inputs[0]
    train_action = inputs[1]
    train_images = inputs[2]

    label_angles = labels[0]
    label_action = labels[1]
    label_target = labels[2]
    label_gripper = labels[3]

    #####################################################
    # Set up variables for the display and for the loop #
    #####################################################

    # Default width is enough to display 8 images. Will scale for more but its not recommended
    width = 128*num_images if 128*num_images > 1024 else 1024
    height = 800
    blank = np.ones((height, width, 3)) * 0.1

    width_offset = 10
    text_offset = 30

    on_disk_height = 0
    formatted_height = 128 + 10
    label_image_height = formatted_height + 128 + 10
    labels_height = label_image_height + 128 + 10

    font = cv2.FONT_HERSHEY_SIMPLEX

    step = 0
    total_steps = len(train_angles)

    while step in range(total_steps):
        display = blank.copy()

        ############################
        # For the top row: on disk #
        ############################
        if pov == 'front':
            tmp_on_disk = on_disk[step].front_rgb
        elif pov == 'wrist':
            tmp_on_disk = on_disk[step].wrist_rgb
        else:
            tmp_on_disk = on_disk[step].front_rgb
        display[on_disk_height:on_disk_height + 128, 0:128, :] = tmp_on_disk  # / 255  # See uint8 note at bottom.

        disk_text = f'On Disk: max={np.max(tmp_on_disk):.3f} min={np.min(tmp_on_disk):.3f}'
        cv2.putText(img=display,
                    text=disk_text,
                    org=(128 + width_offset, on_disk_height + text_offset),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        disk_info = f'(image is blown out because of format used to view formatted and split images)'
        cv2.putText(img=display,
                    text=disk_info,
                    org=(128 + width_offset, on_disk_height + text_offset*2),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        disk_angle = ''
        disk_angle = '[' + ', '.join([disk_angle + f'{p:.3f}' for p in on_disk[step].joint_positions]) + ']'
        disk_angle = f'Joint angles: {disk_angle}'
        cv2.putText(img=display,
                    text=disk_angle,
                    org=(128 + width_offset, on_disk_height + text_offset*3),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        #############################
        # For second row: formatted #
        #############################
        if pov == 'front':
            tmp_formatted = formatted[step].front_rgb
        elif pov == 'wrist':
            tmp_formatted = formatted[step].wrist_rgb
        else:
            tmp_formatted = formatted[step].front_rgb
        display[formatted_height:formatted_height + 128, 0:128, :] = tmp_formatted

        formatted_text = f'Formatted: max={np.max(tmp_formatted):.3f} min={np.min(tmp_formatted):.3f}'
        cv2.putText(img=display,
                    text=formatted_text,
                    org=(128 + width_offset, formatted_height + text_offset),
                    fontFace=font,
                    fontScale=0.75,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        formatted_angle_text = ''
        formatted_angle_text = '[' + \
                               ', '.join([formatted_angle_text + f'{p:.3f}' for p in formatted[step].joint_positions]) \
                               + ']'
        formatted_angle_text = f'Joint angles: {formatted_angle_text}'
        cv2.putText(img=display,
                    text=formatted_angle_text,
                    org=(128 + width_offset, formatted_height + text_offset*2),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        #############################
        # For third row: split data #
        #############################
        for i in range(num_images):
            tmp_input = train_images[step][:, :, 4*i:4*i + 3]
            display[label_image_height:label_image_height + 128, 128*i:128*i + 128, :] = tmp_input

        split_text = f'Split: max={np.max(train_images[step][:, :, 0:4]):.3f} ' \
                     f'min={np.min(train_images[step][:, :, 0:4]):.3f}'
        cv2.putText(img=display,
                    text=split_text,
                    org=(width_offset, labels_height + text_offset),
                    fontFace=font,
                    fontScale=0.75,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        train_action_text = f'Input gripper action: {train_action[step]} (0 := closed, 1 := open))'
        cv2.putText(img=display,
                    text=train_action_text,
                    org=(width_offset, labels_height + text_offset*2),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        train_angle_text = ''
        train_angle_text = '[' + ', '.join([train_angle_text + f'{p:.3f}' for p in train_angles[step]]) + ']'
        train_angle_text = f'Input joint angles: {train_angle_text}'
        cv2.putText(img=display,
                    text=train_angle_text,
                    org=(width_offset, labels_height + text_offset*3),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        label_angle_text = ''
        label_angle_text = '[' + ', '.join([label_angle_text + f'{p:.3f}' for p in label_angles[step]]) + ']'
        label_angle_text = f'Label joint angles: {label_angle_text})'
        cv2.putText(img=display,
                    text=label_angle_text,
                    org=(width_offset, labels_height + text_offset*4),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        label_target_text = ''
        label_target_text = '[' + ', '.join([label_target_text + f'{p:.3f}' for p in label_target[step]]) + ']'
        label_target_text = f'Label target position: {label_target_text})'
        cv2.putText(img=display,
                    text=label_target_text,
                    org=(width_offset, labels_height + text_offset*5),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        label_gripper_text = ''
        label_gripper_text = '[' + ', '.join([label_gripper_text + f'{p:.3f}' for p in label_gripper[step]]) + ']'
        label_gripper_text = f'Label gripper position: {label_gripper_text})'
        cv2.putText(img=display,
                    text=label_gripper_text,
                    org=(width_offset, labels_height + text_offset*6),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
        label_action_text = f'Label gripper action: {label_action[step]} (0 := closed, 1 := open))'
        cv2.putText(img=display,
                    text=label_action_text,
                    org=(width_offset, labels_height + text_offset*7),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ###################
        # Bottom of image #
        ###################
        display_info = f'Demo {ep_num} from {config.possible_data_set[dir_num]}.  ' \
                       f'On step {step} of {total_steps - 1}.'
        cv2.putText(img=display,
                    text=display_info,
                    org=(width_offset, height - text_offset),
                    fontFace=font,
                    fontScale=0.66,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        ################################################################################
        # Display commands and get next input                                          #
        #                                                                              #
        # UINT8 Note...                                                                #
        # Using display.astype('uint8') would allow us to see the on-disk image better #
        # but cause the input images to be essentially blank. The other solution if    #
        # *not* using uint8 is to uncomment line division on line 106 and scale the    #
        # visual display of whats on disk to the range of 0-1                          #
        ################################################################################
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


if __name__ == '__main__':
    main()
