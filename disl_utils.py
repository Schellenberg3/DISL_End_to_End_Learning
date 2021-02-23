from rlbench.backend.const import *
from rlbench.backend.utils import float_array_to_rgb_image
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.backend.utils import image_to_float_array
from rlbench.utils import _resize_if_needed
from PIL import Image
import pickle
import os
from os import listdir
from os.path import join
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

    print(f'loading {example_path}')

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


def _save_demo(_demo, example_path):
    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

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

    for _i, _obs in enumerate(_demo):
        left_shoulder_rgb = Image.fromarray(
            (_obs.left_shoulder_rgb * 255).astype(np.uint8))
        left_shoulder_depth = float_array_to_rgb_image(
            _obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (_obs.left_shoulder_mask * 255).astype(np.uint8)).convert('RGB')
        right_shoulder_rgb = Image.fromarray(
            (_obs.right_shoulder_rgb * 255).astype(np.uint8))
        right_shoulder_depth = float_array_to_rgb_image(
            _obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (_obs.right_shoulder_mask * 255).astype(np.uint8)).convert('RGB')

        wrist_rgb = Image.fromarray((_obs.wrist_rgb * 255).astype(np.uint8))
        wrist_depth = float_array_to_rgb_image(
            _obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((_obs.wrist_mask * 255).astype(np.uint8)).convert('RGB')

        front_rgb = Image.fromarray((_obs.front_rgb * 255).astype(np.uint8))
        front_depth = float_array_to_rgb_image(
            _obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((_obs.front_mask * 255).astype(np.uint8)).convert('RGB')

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % _i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % _i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % _i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % _i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % _i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % _i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % _i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % _i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % _i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % _i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % _i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % _i))

        # We save the images separately, so set these to None for pickling.
        _obs.left_shoulder_rgb = None
        _obs.left_shoulder_depth = None
        _obs.left_shoulder_mask = None
        _obs.right_shoulder_rgb = None
        _obs.right_shoulder_depth = None
        _obs.right_shoulder_mask = None
        _obs.wrist_rgb = None
        _obs.wrist_depth = None
        _obs.wrist_mask = None
        _obs.front_rgb = None
        _obs.front_depth = None
        _obs.front_mask = None

    num_steps = len(_demo)

    if not (num_steps == len(listdir(left_shoulder_rgb_path))):
        print(f'[WARN] Broken dataset assumption. This file may not load properly. '
              f'len(_demo)={num_steps} != len(left_shoulder_rgb)={len(listdir(left_shoulder_rgb_path))}')

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as file:
        pickle.dump(_demo, file)