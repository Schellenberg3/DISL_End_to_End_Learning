from rlbench.backend.const import *
from rlbench.backend.utils import float_array_to_rgb_image
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.backend.utils import image_to_float_array
from rlbench.utils import _resize_if_needed
from PIL import Image
from custom_networks import rnn_position_vision
from custom_networks import rnn_position_vision_4
from custom_networks import position_vision
from custom_networks import rnn_vision
from os import listdir
from os.path import join
import pickle
import os
import shutil
import numpy as np
import random


# todo: document + type assist each function
def check_and_make(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)


def get_order(number, epochs=1):
    order = []
    for e in range(epochs):
        order += random.sample(list(range(number)), number)
    return order


def save_demos(demos, path, start_episode=0):
    for i, demo in enumerate(demos):
        p = path + '/variation0/episodes/episode%d' % (i + start_episode)
        _save_demo(demo, p)
        #print(f'[Info] Saved demo {i + start_episode} at location: {p}')


def load_data(path, example_num, obs_config):

    # check if path exists

    example_path = path + '/episode%d' % example_num

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


def format_demos(all_demos):
    for _demo in range(len(all_demos)):
        for _step in range(len(all_demos[_demo])):
            all_demos[_demo][_step].front_rgb = all_demos[_demo][_step].front_rgb / 255
            all_demos[_demo][_step].left_shoulder_rgb = all_demos[_demo][_step].left_shoulder_rgb / 255
            all_demos[_demo][_step].right_shoulder_rgb = all_demos[_demo][_step].right_shoulder_rgb / 255
            all_demos[_demo][_step].wrist_rgb = all_demos[_demo][_step].wrist_rgb / 255

            # If we decide to use mask: revisit and confirm that this rescales the values.
            all_demos[_demo][_step].front_mask = all_demos[_demo][_step].front_mask / 255
            all_demos[_demo][_step].left_shoulder_mask = all_demos[_demo][_step].left_shoulder_mask / 255
            all_demos[_demo][_step].right_shoulder_mask = all_demos[_demo][_step].right_shoulder_mask / 255
            all_demos[_demo][_step].wrist_mask = all_demos[_demo][_step].wrist_mask / 255

    return all_demos


def format_data(demo):
    for step in range(len(demo)):
        demo[step].front_rgb = demo[step].front_rgb / 255
        demo[step].left_shoulder_rgb = demo[step].left_shoulder_rgb / 255
        demo[step].right_shoulder_rgb = demo[step].right_shoulder_rgb / 255
        demo[step].wrist_rgb = demo[step].wrist_rgb / 255
        demo[step].front_mask = demo[step].front_mask / 255
        demo[step].left_shoulder_mask = demo[step].left_shoulder_mask / 255
        demo[step].right_shoulder_mask = demo[step].right_shoulder_mask / 255
        demo[step].wrist_mask = demo[step].wrist_mask / 255

    return demo


def split_data(demo):
    data = []
    images = []
    label = []

    for step in range(len(demo)):
        data.append(np.append(demo[step].joint_positions,
                               demo[step].gripper_open))
        images.append(np.dstack((demo[step].front_rgb,
                                 demo[step].front_depth)))
        try:
            label.append(np.append(demo[step + 1].joint_positions,
                                    demo[step + 1].gripper_open))
        except IndexError:
            label.append(np.append(demo[step].joint_positions,
                                    demo[step].gripper_open))

    return data, images, label


def split_data_4(demo):
    blank_image = np.zeros((128, 128, 4))

    data = []
    images = []
    label = []

    for step in range(len(demo)):
        data.append(np.append(demo[step].joint_positions,
                              demo[step].gripper_open))
        im_array = np.dstack((demo[step].front_rgb,
                              demo[step].front_depth))

        for i in [1, 2, 3]:
            if step - i < 0:
                im_array = np.dstack((im_array, blank_image))
            else:
                im_array = np.dstack((im_array,
                                     np.dstack((demo[step - i].front_rgb,
                                                demo[step - i].front_depth))))

        images.append(im_array)

        try:
            label.append(np.append(demo[step + 1].joint_positions,
                                   demo[step + 1].gripper_open))
        except IndexError:
            label.append(np.append(demo[step].joint_positions,
                                   demo[step].gripper_open))

    return data, images, label


def split_demos(all_demos, train_split, num_demos):
    _train_data = []
    _train_images = []
    _train_label = []
    _test_data = []
    _test_images = []
    _test_label = []
    data = []
    images = []
    label = []

    _num = 0
    _num_train = 0
    _num_test = 0
    _num_demos = len(all_demos)

    training_demos = np.random.choice(range(_num_demos),
                                      round(train_split*num_demos), replace=False)

    for _demo in range(num_demos):
        _num = 0
        data = []
        images = []
        label = []

        for _step in range(len(all_demos[_demo])):
            data.append(np.append(all_demos[_demo][_step].joint_positions,
                                  [all_demos[_demo][_step].gripper_open]))
            images.append(np.dstack((all_demos[_demo][_step].front_rgb,
                                    all_demos[_demo][_step].front_depth)))
            try:
                label.append(np.append(all_demos[_demo][_step + 1].joint_positions,
                                       [all_demos[_demo][_step + 1].gripper_open]))
            except IndexError:
                label.append(np.append(all_demos[_demo][_step].joint_positions,
                                       [all_demos[_demo][_step].gripper_open]))
            _num += 1

        if _demo in training_demos:
            _train_data.append(data)
            _train_images.append(images)
            _train_label.append(label)
            _num_train += _num
        else:
            _test_data.append(data)
            _test_images.append(images)
            _test_label.append(label)
            _num_test += _num

    print(f"\n---> Split data into {len(training_demos)} training demos ({_num_train} samples)"
          f" and {_num_demos - len(training_demos)} testing demos ({_num_test} samples).\n")

    return _train_data, _train_images, _train_label, _num_train, _test_data, _test_images, _test_label, _num_test


def _save_demo(demo, episode_path):
    """
    Takes one full episode and saves it in the

    :param demo: List[Observations]
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

    for i, obs in enumerate(demo):
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

    num_steps = len(demo)

    if not (num_steps == len(listdir(left_shoulder_rgb_path))):
        print(f'[WARN] Broken dataset assumption. This file may not load properly. '
              f'len(_demo)={num_steps} != len(left_shoulder_rgb)={len(listdir(left_shoulder_rgb_path))}')

    # Save the low-dimension data
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'wb') as file:
        pickle.dump(demo, file)


def check_yes(text):
    """
    Verifies that a users input is either yes or 1

    :param text: Prompt to get input
    :return: bool
    """
    response = input(text)
    if response in ['y', 'Y', 'yes', 'Yes', 'YES', '1', 1]:
        return True
    else:
        return False


def format_time(seconds):
    """
    Takes an amount of seconds and returns a string formatted into hours,
    minutes, and seconds

    :param seconds: Time as a float
    :return: str
    """
    h = int(seconds/3600)
    m = int((seconds - h*3600)/60)
    s = (seconds - h*3600 - m*60)
    return f'{h:3.0f}h{m:3.0f}m{s:3.0f}s'


# todo consider moving to separate files
#    - utils and maybe custom_networks go to assets?
#    - use if __name__ == 'imitation_learner' to import TensorFlow only when needed? (NameError if module isn't defined)
class EndToEndConfig:

    def __init__(self):
        # DO NOT MOVE THIS FILE FROM THE MAIN FOLDER. WILL BREAK DIRECTORY LOCATION ASSUMPTION
        self.network_root = join(os.path.dirname(os.path.realpath(__file__)), 'trained_networks')
        self.data_root = join(os.path.dirname(os.path.realpath(__file__)), 'data')
        self.custom_networks = {"position_vision": ("pv",
                                                    position_vision,
                                                    split_data),
                                "rnn_vision": ("rnn-v",
                                               rnn_vision,
                                               split_data),
                                "rnn_position_vision": ("rnn-pv",
                                                        rnn_position_vision,
                                                        split_data),
                                "rnn_position_vision_4": ("rnn-pv4",
                                                          rnn_position_vision_4,
                                                          split_data_4),
                                }

    # todo update reference in imitation_learner
    def get_new_network(self):
        """
        Lists network options from custom_networks and lets user choose pick a
        configuration. Returns the network's name, the network function, and
        supporting data split function.

        :return: Tuple[str, Model, function]
        """

        print('\nThe following networks are available:')
        list_keys = []
        for i, (k, v) in enumerate(self.custom_networks.items()):
            print(f'Option {i}.....{k}')
            list_keys.append(k)

        try:
            model_selection = int(input('\nPlease enter the option # for the network you would like to create: '))
            if model_selection > len(list_keys):
                exit('[ERROR] Selections must be integers. Exiting program')
        except ValueError:
            exit('[ERROR] Selections must be integers. Exiting program')

        if check_yes('\nWill a data set with domain randomization be used in training? (y/n): '):
            domain = "rand"
        else:
            domain = "norm"

        cnn_setting = input('\nEnter 0 to use a James inspired CNN, 1 to use a Hermann inspired CNN, or '
                            'anything else to use the custom one: ')
        if cnn_setting == '0':
            cnn_setting = 'James'
        elif cnn_setting == '1':
            cnn_setting = "Hermann"
        else:
            cnn_setting = "Custom"

        key = list_keys[model_selection]
        name = f'{self.custom_networks[key][0]}_{domain}_{cnn_setting}'
        model = self.custom_networks[key][1](cnn_setting)
        split = self.custom_networks[key][2]

        print(f'\n[Info] Network will be {key} configured with {cnn_setting} CNN')

        return name, model, split

    # todo write code to get a saved model

    def set_directories(self):
        """
        Lists the directories the data/ and asks the user to pick which
        on to use for testing and training

        :return: Tuple[train_dir, test_dir]
        """
        possible_data_set = []
        i = 0  # not enumerate -only count if the item in dataset_root is a folder with children
        print(f'\nThe data from the following directories may be used: ')
        for folder in listdir(join(self.data_root)):
            for data in listdir(join(self.data_root, folder)):
                possible_data_set.append(join(folder, data))
                try:
                    num = len(listdir(join(self.data_root, possible_data_set[i], 'variation0', 'episodes')))
                except FileNotFoundError:
                    num = 'NONE'
                print('{:.<20s}{:.<20s}{:.<5s}'.format(f'Directory {i}',
                                                       f'{num} episodes',
                                                       f'{possible_data_set[i]}'))
                i += 1

        try:
            train_num = int(input('\nEnter directory # for training: '))
            test_num = int(input('Enter directory # for testing: '))

            if train_num == test_num:
                exit('\n[ERROR] Cannot test and train on the same directory. Exiting program.')
            elif train_num < 0 or test_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            train_dir = join(self.data_root, possible_data_set[train_num], 'variation0', 'episodes')
            test_dir = join(self.data_root, possible_data_set[test_num], 'variation0', 'episodes')

            return train_dir, test_dir
        except (ValueError, IndexError) as e:
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def get_episode_amounts(self, train_dir, test_dir):
        """
        Assists in getting and checking the number of training and testing demos to use
        and gets the number of training epochs

        :param train_dir: full directory to training episodes
        :param test_dir: full directory to testing episodes
        :return: Tuple[num_test, num_train, epochs
        """
        text = ['training', 'testing']
        amounts = [0, 0]
        for i, d in enumerate([train_dir, test_dir]):
            available = len(listdir(d))
            print(f'\nThere are {available} episodes available for {text[i]} at {d}')
            to_use = int(input('Enter how many to use (or -1 for all): '))
            if to_use > available or to_use <= -1:
                print(f'[Info] Using all {available} {text[i]} episodes.')
                to_use = available
            amounts[i] = to_use

        epochs = int(input('\nEnter how many epochs use: '))
        if epochs < 1:
            print(f'[Info] Setting epochs to 1')
            epochs = 1

        return amounts, epochs
