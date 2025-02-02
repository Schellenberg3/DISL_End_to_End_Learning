from rlbench.demo import Demo
from rlbench.backend.const import *
from rlbench.backend.utils import float_array_to_rgb_image
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.backend.utils import image_to_float_array
from rlbench.observation_config import ObservationConfig
from rlbench.utils import _resize_if_needed

from os.path import join
from os import listdir

from typing import List
from typing import Tuple
from typing import Union

from PIL import Image

import numpy as np

import pickle
import shutil
import random
import os
import re


# This file contains many common functions used by code in the DISL End-to-End project
# Broadly, these code modules can be separated into the following categories...
#    - General functions that aren't specific to this project
#    - Functions for saving RLBench episodes to disk and loading them to python objects
#    - Functions for modifying the data in an RLBench episode object
#
# Code is grouped in these categories and the docstring describes how to used each function.

#####################
# General functions #
#####################

def alpha_numeric_sort(unsorted: List[str]) -> List[str]:
    """ Sorts a list by alphabetical order and accounts for numeric values

    :param unsorted: the unsorted list of words

    :return: the sorted list
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(unsorted, key=alphanum_key)


def check_yes(text: str) -> bool:
    """ Verifies that a users input is either yes or 1

    :param text: Prompt to get input

    :return: bool
    """
    response = input(text)
    if response in ['y', 'Y', 'yes', 'Yes', 'YES', '1', 1]:
        return True
    else:
        return False


def format_time(seconds: float) -> str:
    """ Takes an amount of seconds and returns a string formatted into hours,
    minutes, and seconds.

    :param seconds: Time as a float

    :return: String with the time formatted
    """
    h = int(seconds/3600)
    m = int((seconds - h*3600)/60)
    s = (seconds - h*3600 - m*60)
    return f'{h:3.0f}h{m:3.0f}m{s:3.0f}s'


def check_and_make(directory: str) -> None:
    """ Take a path to a directory. If the path doesn't
    exist the directory is created. If it does, the directories
    are removed first.

    :param directory: Path to a directory

    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)


def get_order(amount: int, available: int, epochs=1, ordered: bool=False) -> List[int]:
    """ Used for selecting episode numbers from the datasets for testing and
    training. First picks amount from available. Then for each epoch
    randomly shuffles the selection. Finally, the selection is returned
    as a list.

    :param amount:    Number of episodes to pick
    :param available: The total number of episodes to pick from
    :param epochs:    How many times the selected episodes should appear
    :param ordered:   If true returns first N episodes where N is the amount specified
                      This is most useful for evaluation of a trained network.

    :return: list of episode numbers
    """
    order = []
    if not ordered:
        to_pick_from = random.sample(list(range(available)), amount)

        for e in range(epochs):
            order += random.sample(to_pick_from, amount)
    else:
        for e in range(epochs):
            order += list(range(amount))

    return order


#################################################
# Functions for saving and loading RLBench data #
#################################################

def save_episodes(episodes: List[Demo], data_set_path: str, start_episode=0) -> None:
    """ Takes a list of demos/episodes and saves them to disk under the
    data folder.

    :param episodes:      An array of type RLBench Demo objects
    :param data_set_path: Path to the data set's root, usually the task's name
    :param start_episode: Offset to save at if there are existing demos

    :return: None
    """
    for i, episode in enumerate(episodes):
        p = join(data_set_path,
                 'variation0',
                 'episodes',
                 f'episode{i + start_episode}')
        _save_episode(episode, p)


def _save_episode(episode: Demo, episode_path: str) -> None:
    """ Takes one full demo/episode and saves it in the provided
    directory.

    :param episode:      A single RLBench episode (Demo object), also a list of observations
    :param episode_path: directory to save the example in

    :return: None
    """
    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(episode_path,
                                          LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(episode_path,
                                            LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(episode_path,
                                           LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(episode_path,
                                           RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(episode_path,
                                             RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(episode_path,
                                            RIGHT_SHOULDER_MASK_FOLDER)
    wrist_rgb_path = os.path.join(episode_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(episode_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(episode_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(episode_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(episode_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(episode_path, FRONT_MASK_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(episode):
        left_shoulder_rgb = Image.fromarray(
            (obs.left_shoulder_rgb * 255).astype(np.uint8))
        left_shoulder_depth = float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8)).convert('RGB')
        right_shoulder_rgb = Image.fromarray(
            (obs.right_shoulder_rgb * 255).astype(np.uint8))
        right_shoulder_depth = float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8)).convert('RGB')

        wrist_rgb = Image.fromarray((obs.wrist_rgb * 255).astype(np.uint8))
        wrist_depth = float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8)).convert('RGB')

        front_rgb = Image.fromarray((obs.front_rgb * 255).astype(np.uint8))
        front_depth = float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8)).convert('RGB')

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_mask = None

    num_steps = len(episode)

    if not (num_steps == len(listdir(left_shoulder_rgb_path))):
        print(f'[WARN] Broken dataset assumption. This file may not load properly. '
              f'len(_demo)={num_steps} != len(left_shoulder_rgb)={len(listdir(left_shoulder_rgb_path))}')

    # Save the low-dimension data
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'wb') as file:
        pickle.dump(episode, file)


def load_data(path: str, episode_num: int, obs_config: ObservationConfig) -> Demo:
    """ Loads a full demo/episode from disk based on the provided
    data path, episode number, and observation configuration

    :param path:        Data set directory
    :param episode_num: Requested episode number
    :param obs_config:  RLBench observation configuration

    :return: Demo object for the requested episode
    """
    example_path = join(path, f'episode{episode_num}')

    with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
        obs = pickle.load(f)

    l_sh_rgb_f = join(example_path, LEFT_SHOULDER_RGB_FOLDER)
    l_sh_depth_f = join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    l_sh_mask_f = join(example_path, LEFT_SHOULDER_MASK_FOLDER)
    r_sh_rgb_f = join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
    r_sh_depth_f = join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    r_sh_mask_f = join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
    wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
    front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
    front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_f = join(example_path, FRONT_MASK_FOLDER)

    num_steps = len(obs)

    if not (num_steps == len(listdir(l_sh_rgb_f)) == len(
            listdir(l_sh_depth_f)) == len(listdir(r_sh_rgb_f)) == len(
            listdir(r_sh_depth_f)) == len(listdir(wrist_rgb_f)) == len(
            listdir(wrist_depth_f)) == len(listdir(front_rgb_f)) == len(
            listdir(front_depth_f))):
        raise RuntimeError('Broken dataset assumption')

    for i in range(num_steps):
        si = IMAGE_FORMAT % i
        if obs_config.left_shoulder_camera.rgb:
            obs[i].left_shoulder_rgb = join(l_sh_rgb_f, si)
        if obs_config.left_shoulder_camera.depth:
            obs[i].left_shoulder_depth = join(l_sh_depth_f, si)
        if obs_config.left_shoulder_camera.mask:
            obs[i].left_shoulder_mask = join(l_sh_mask_f, si)
        if obs_config.right_shoulder_camera.rgb:
            obs[i].right_shoulder_rgb = join(r_sh_rgb_f, si)
        if obs_config.right_shoulder_camera.depth:
            obs[i].right_shoulder_depth = join(r_sh_depth_f, si)
        if obs_config.right_shoulder_camera.mask:
            obs[i].right_shoulder_mask = join(r_sh_mask_f, si)
        if obs_config.wrist_camera.rgb:
            obs[i].wrist_rgb = join(wrist_rgb_f, si)
        if obs_config.wrist_camera.depth:
            obs[i].wrist_depth = join(wrist_depth_f, si)
        if obs_config.wrist_camera.mask:
            obs[i].wrist_mask = join(wrist_mask_f, si)
        if obs_config.front_camera.rgb:
            obs[i].front_rgb = join(front_rgb_f, si)
        if obs_config.front_camera.depth:
            obs[i].front_depth = join(front_depth_f, si)
        if obs_config.front_camera.mask:
            obs[i].front_mask = join(front_mask_f, si)

        # Remove low dim info if necessary
        if not obs_config.joint_velocities:
            obs[i].joint_velocities = None
        if not obs_config.joint_positions:
            obs[i].joint_positions = None
        if not obs_config.joint_forces:
            obs[i].joint_forces = None
        if not obs_config.gripper_open:
            obs[i].gripper_open = None
        if not obs_config.gripper_pose:
            obs[i].gripper_pose = None
        if not obs_config.gripper_joint_positions:
            obs[i].gripper_joint_positions = None
        if not obs_config.gripper_touch_forces:
            obs[i].gripper_touch_forces = None
        if not obs_config.task_low_dim_state:
            obs[i].task_low_dim_state = None

    for i in range(num_steps):
        if obs_config.left_shoulder_camera.rgb:
            obs[i].left_shoulder_rgb = np.array(
                _resize_if_needed(
                    Image.open(obs[i].left_shoulder_rgb),
                    obs_config.left_shoulder_camera.image_size))
        if obs_config.right_shoulder_camera.rgb:
            obs[i].right_shoulder_rgb = np.array(
                _resize_if_needed(
                    Image.open(obs[i].right_shoulder_rgb),
                    obs_config.right_shoulder_camera.image_size))
        if obs_config.wrist_camera.rgb:
            obs[i].wrist_rgb = np.array(
                _resize_if_needed(
                    Image.open(obs[i].wrist_rgb),
                    obs_config.wrist_camera.image_size))
        if obs_config.front_camera.rgb:
            obs[i].front_rgb = np.array(
                _resize_if_needed(
                    Image.open(obs[i].front_rgb),
                    obs_config.front_camera.image_size))

        if obs_config.left_shoulder_camera.depth:
            obs[i].left_shoulder_depth = image_to_float_array(
                _resize_if_needed(
                    Image.open(obs[i].left_shoulder_depth),
                    obs_config.left_shoulder_camera.image_size),
                    DEPTH_SCALE)
        if obs_config.right_shoulder_camera.depth:
            obs[i].right_shoulder_depth = image_to_float_array(
                _resize_if_needed(
                    Image.open(obs[i].right_shoulder_depth),
                    obs_config.right_shoulder_camera.image_size),
                    DEPTH_SCALE)
        if obs_config.wrist_camera.depth:
            obs[i].wrist_depth = image_to_float_array(
                _resize_if_needed(
                    Image.open(obs[i].wrist_depth),
                    obs_config.wrist_camera.image_size),
                    DEPTH_SCALE)
        if obs_config.front_camera.depth:
            obs[i].front_depth = image_to_float_array(
                _resize_if_needed(
                    Image.open(obs[i].front_depth),
                    obs_config.front_camera.image_size),
                    DEPTH_SCALE)

        # Masks are stored as coded RGB images.
        # Here we transform them into 1 channel handles.
        if obs_config.left_shoulder_camera.mask:
            obs[i].left_shoulder_mask = rgb_handles_to_mask(
                np.array(_resize_if_needed(Image.open(
                    obs[i].left_shoulder_mask),
                    obs_config.left_shoulder_camera.image_size)))
        if obs_config.right_shoulder_camera.mask:
            obs[i].right_shoulder_mask = rgb_handles_to_mask(
                np.array(_resize_if_needed(Image.open(
                    obs[i].right_shoulder_mask),
                    obs_config.right_shoulder_camera.image_size)))
        if obs_config.wrist_camera.mask:
            obs[i].wrist_mask = rgb_handles_to_mask(np.array(
                _resize_if_needed(Image.open(
                    obs[i].wrist_mask),
                    obs_config.wrist_camera.image_size)))
        if obs_config.front_camera.mask:
            obs[i].front_mask = rgb_handles_to_mask(np.array(
                _resize_if_needed(Image.open(
                    obs[i].front_mask),
                    obs_config.front_camera.image_size)))

    return obs


################################################
# Functions for modifying RLBench episode data #
################################################

def format_data(episode: Demo, pov: Union[List[str], str]) -> Demo:
    """ Takes a demo/episode loaded from disk and normalizes the images to
    a range of [0,1]. Also scales the joint positions from [-3.14, 3.14]
    to [0,1] to normalize.

    :param episode: Input episode
    :param pov:     List of the POV to format, e.g. ['front', 'wrist', 'right_shoulder', 'right_shoulder']
                    would format each image but ['front'] only formats the front image.

    :return: Same demonstration, now formatted
    """

    if type(pov) == str:
        pov = [pov]

    front = True if 'front' in pov else False
    wrist = True if 'wrist' in pov else False
    left_shoulder = True if 'left_shoulder' in pov else False
    right_shoulder = True if 'right_shoulder' in pov else False

    for step in range(len(episode)):
        if front:
            episode[step].front_rgb = episode[step].front_rgb / 255
        if left_shoulder:
            episode[step].left_shoulder_rgb = episode[step].left_shoulder_rgb / 255
        if right_shoulder:
            episode[step].right_shoulder_rgb = episode[step].right_shoulder_rgb / 255
        if wrist:
            episode[step].wrist_rgb = episode[step].wrist_rgb / 255

        episode[step].joint_positions = scale_panda_pose(episode[step].joint_positions, 'down')

    return episode


def scale_pose(array: np.ndarray, old_min=0., old_max=1., new_min=-3.14, new_max=3.14) -> np.ndarray:
    """ Scales all values of an array from one range to another. By default this is from [0,1]
    to [-3.14, 3.14].  Used to normalize position values in training.  When using a network this
    should be called on the position (but not gripper!) part of the output.

    :param array:   Old values
    :param old_min: Old starting value
    :param old_max: Old ending value
    :param new_min: New starting value
    :param new_max: New ending value

    :return: New values
    """
    for i in range(len(array)):
        array[i] = scale_value(array[i], old_min=old_min, old_max=old_max, new_min=new_min, new_max=new_max)
    return array


def scale_value(value, old_min=0., old_max=1., new_min=-3.14, new_max=3.14) -> float:
    """
    Linearly scales a single value from [old_min, old_max] to [new_min, new_max]

    Used as a helper for scale_pose and scale_panda_pose.

    :param value:   Old value
    :param old_min: Old starting value
    :param old_max: Old ending value
    :param new_min: New starting value
    :param new_max: New ending value

    :return: New value
    """
    return (new_max - new_min)*(value - old_max)/(old_max - old_min) + new_max


def scale_pose_up(array: np.ndarray) -> np.ndarray:
    """
    Scales each element of the input array from [0, 1] up to [-3.14, 3.14]

    :param array: Array to be scaled
    """
    return scale_pose(array, 0, 1, -3.14, 3.14)


def scale_pose_down(array: np.ndarray) -> np.ndarray:
    """
    Scales each element of the input array from [-3.14, 3.14] down to [0, 1]

    :param array: Array to be scaled
    """
    return scale_pose(array, -3.14, 3.14, 0, 1)


def scale_panda_pose(array: np.ndarray, direction: str = 'up') -> np.ndarray:
    """
    Uses the panda joint angles found in CoppeliaSim and the scale_pose function to
    linearly scale the joint values to and from a range of [0, 1]

    :param array:     Input array to be scaled
    :param direction: Either 'up' to return to RLBench values from [0, 1] or 'down' to
                      retrieve normalized values from RLBench values.

    :return: Scaled array of the same dimensions
    """
    # Joint [min, max] in degrees from CoppeliaSim
    panda = np.array([[-166, 332],
                      [-101, 202],
                      [-166, 332],
                      [-176, 172],
                      [-166, 332],
                      [-1, 216],
                      [-166, 332]])
    panda = np.deg2rad(panda)

    for i, val in enumerate(panda):
        if direction == 'up':
            old_min = 0.
            old_max = 1.
            new_min = val[0]
            new_max = val[1]
        else:  # if direction == 'down'
            old_min = val[0]
            old_max = val[1]
            new_min = 0.
            new_max = 1.
        array[i] = scale_value(array[i], old_min=old_min, old_max=old_max, new_min=new_min, new_max=new_max)
    return array


def step_images(image_list: List[np.ndarray], new_image: np.ndarray) -> List[np.ndarray]:
    """
    Takes a list (or 'history') of images and adds a new images to the front while passing
    the previous images back. Returns this new list.

    :param image_list: List of images from previous step
    :param new_image:  Image to add at current step

    :return: List of images for current step
    """
    for i in (range(len(image_list) - 1, 0, -1)):
        image_list[i] = image_list[i - 1]
    image_list[0] = new_image

    return image_list


def blank_image_list(num_images: int) -> List[np.ndarray]:
    """
    Creates a list of blank (all zero) depth images. Each image is a numpy array of size 128x128x4.

    :param num_images: Number of blank images to initialize the list with.

    :return: List of blank images
    """
    images = []
    blank_image = np.zeros((128, 128, 4))
    for i in range(num_images):
        images.append(blank_image)
    return images


def get_wrist_rgbd(episode: Demo, step: int) -> np.ndarray:
    """
    Getter for an episodes wrist image as 128x128x4 matrix. Called by split_data().

    :param episode: Episode to load a wrist image from
    :param step:    Time step of image
    """
    return np.dstack((episode[step].wrist_rgb,
                      episode[step].wrist_depth))


def get_front_rgbd(episode: Demo, step: int) -> np.ndarray:
    """
    Getter for an episodes front image as 128x128x4 matrix. Called by split_data().

    :param episode: Episode to load a front image from
    :param step:    Time step of image
    """
    return np.dstack((episode[step].front_rgb,
                      episode[step].front_depth))


def split_data(episode: Demo, num_images: int = 4, pov: str = 'front', predict_mode: str = 'positions') -> \
        Tuple[List[List], List[List]]:
    """ Takes an episode and splits it into the joint data (including gripper), the depth image,
    and the next position (ground truth label). Returns a list with the values for each
    of these at evey step in the episode.

    :param episode:      Episode to split
    :param num_images:   Number of images to use for the image input.
    :param pov:          Either 'wrist' or 'front', tells which images to use
    :param predict_mode: What mode to use for the output joint prediction. Either 'positions' or 'velocities'

    :return: Tuple of lists for joint data, depth image, and ground truth label
    """
    predict_mode = f'joint_{predict_mode}'  # format to RLBench's Observation object's member variable name convention

    # Input Data
    angles = []  # Input angle
    action = []  # Input gripper action (open or close)
    images = []  # Input images
    image_list = blank_image_list(num_images)  # Helper for creating the images input at each step

    # Prediction/Ground Truth labels
    label_joints = []   # True angles or velocities
    label_action = []   # True gripper action (open or close)
    label_target = []   # True position of the target object (e.g. a cup)
    label_gripper = []  # True position of the robot's gripper

    if pov == 'wrist':
        get_image = get_wrist_rgbd
    elif pov == 'front':
        get_image = get_front_rgbd
    else:
        print(f'Invalid point ov view in split data')
        get_image = get_front_rgbd

    for step in range(len(episode)):

        image = get_image(episode, step)
        image_list = step_images(image_list=image_list, new_image=image)
        image_stack = np.dstack(image_list)

        #########################################################
        # Inputs                                                #
        # - Current image set, gripper action, and joint angles #
        #########################################################

        images.append(image_stack)
        angles.append(episode[step].joint_positions)
        action.append(episode[step].gripper_open)

        ##################################################
        # Labels                                         #
        # - Current target position and gripper position #
        # - Next joint angles and gripper action         #
        ##################################################

        # TODO: Possible future update to this section...
        #       The dataset records (X,Y,Z,Qx,Qy,Qz,Qw) but we only want (X,Y,Z) for now
        label_target.append(episode[step].task_low_dim_state[0][:3])
        label_gripper.append(episode[step].task_low_dim_state[-1][:3])
        array_action = np.zeros(2)
        try:
            label_joints.append(getattr(episode[step + 1], predict_mode))
            next_action = episode[step + 1].gripper_open
        except IndexError:
            label_joints.append(getattr(episode[step], predict_mode))
            next_action = episode[step].gripper_open
        array_action[int(next_action)] = 1
        label_action.append(array_action.copy())

    inputs = [angles, action, images]
    labels = [label_joints, label_action, label_target,label_gripper]

    return inputs, labels


def get_data(episode_dir: str, episode_num: int, obs_config: ObservationConfig, pov: Union[str, List[str]],
             num_images: int, predict_mode: str) -> Tuple[List[List], List[List]]:
    """
    Convenience function that calls load, split, then format to return the inputs and labels for a training
    episode.

    :param episode_dir:  Which directory to load an episode from
    :param episode_num:  Which episode to load from the directory
    :param obs_config:   RLBench ObservationConfiguration
    :param pov:          Either 'wrist' or 'front', tells which images to use
    :param num_images:   Number of images to use for the image input.
    :param predict_mode: What mode to use for the output joint prediction. Either 'positions' or 'velocities'

    :return: Tuple of lists for joint data, depth image, and ground truth label
    """
    return split_data(format_data(load_data(episode_dir,
                                            episode_num,
                                            obs_config=obs_config
                                            ),
                                  pov=pov
                                  ),
                      num_images=num_images,
                      pov=pov,
                      predict_mode=predict_mode)

