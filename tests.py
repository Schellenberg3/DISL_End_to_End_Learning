from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.environment import Environment
from rlbench.tasks import DislPickUpBlueCup
from rlbench.demo import Demo

from utils.utils import save_episodes
from utils.utils import load_data
from utils.utils import format_data
from utils.utils import split_data
from utils.utils import scale_pose_up
from utils.utils import scale_pose_down
from utils.utils import blank_image_list
from utils.utils import step_images

from copy import deepcopy

from os.path import isdir
from os.path import join
from os import listdir
from os import getcwd

from typing import Tuple
from typing import Union
from typing import List

import numpy as np

import pytest


########################################################################################
# Testing for functions in utils.utils that are used throughout the end-to-end project #
#                                                                                      #
# Using RLBench, two demos of the disl_pick_up_blue_cup task are generated each time   #
# this test module is run. The copies of the data from these runs is used in multiple  #
# tests in the file. See the data_input test feature.                                  #
#                                                                                      #
# Because we generate new data for each run and do hundreds of array comparisons the   #
# expected run time for this test module is around 1 minute.                           #
########################################################################################

@pytest.mark.parametrize("input_array, result_array", [
    (np.ones(1), np.array([3.14])),
    (np.ones(3), np.array([3.14, 3.14, 3.14])),
    (np.zeros(1), np.array([-3.14])),
    (np.zeros(3), np.array([-3.14, -3.14, -3.14])),
    (0.5*np.ones(1), np.array([0.])),
    (0.5*np.ones(3), np.array([0., 0., 0.])),
    (np.array([1, 0.5, 0.]), np.array([3.14, 0., -3.14])),
    (np.array([2, -2]), np.array([6.28, -6.28])),
])
def test_scale_pose_up(input_array: np.ndarray, result_array: np.ndarray) -> None:
    """
    Tests the scale_pose_up function which is a convenience function for scale pose
    that takes values from [0, 1] to [-3.14, 3.14]

    :param input_array:  Input array to be scaled up
    :param result_array: Comparison 'true' array
    """
    assert scale_pose_up(input_array).all() == result_array.all()


@pytest.mark.parametrize("input_array, result_array", [
    (3.14*np.ones(1), np.array([1.])),
    (3.14*np.ones(3), np.array([1., 1., 1.])),
    (np.zeros(1), np.array([0.5])),
    (np.zeros(3), np.array([0.5, 0.5, 0.5])),
    (-3.14*np.ones(1), np.array([0.])),
    (-3.14*np.ones(3), np.array([0., 0., 0.])),
    (np.array([3.14, 0., -3.14]), np.array([1, 0.5, 0.])),
    (np.array([6.28, -6.28]), np.array([2, -1])),
])
def test_scale_pose_down(input_array: np.ndarray, result_array: np.ndarray) -> None:
    """
    Tests the scale_pose_down function which is a convenience function for scale pose
    that takes values from [-3.14, 3.14] to [0, 1]

    :param input_array:  Input array to be scaled up
    :param result_array: Comparison 'true' array
    """
    assert scale_pose_down(input_array).all() == result_array.all()


################################################################
# Constants used in test_blank_image_list and test_step_images #
################################################################
BLANK = np.zeros((128, 128, 4))
ONES = np.ones((128, 128, 4))


@pytest.mark.parametrize("number, actual, expected", [
    (8, [BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK], True),
    (4, [BLANK, BLANK, BLANK, BLANK], True),
    (1, [BLANK], True),
    (0, [], True),
])
def test_blank_image_list(number: int, actual: List[np.ndarray], expected: bool) -> None:
    """
    Tests blank_image_list which creates a list of 128x128x4 images initialized to zero

    :param number:   The amount of images in the desired list
    :param actual:   The actual list that is desired
    :param expected: The expected result of the test, success or failure
    """
    blank_list = blank_image_list(number)
    result = [(blank_list[i] == actual[i]).all() for i in range(number)]
    result = False if False in result else True

    assert result == expected


@pytest.mark.parametrize("steps, num_images, inputs, actual, stack, expected", [
    (3, 2, [ONES, 2*ONES, 3*ONES], [3*ONES, 2*ONES], np.dstack((3*ONES, 2*ONES)), True),
    (3, 2, [ONES, 2*ONES, 3*ONES], [2*ONES, 3*ONES], np.dstack((2*ONES, 3*ONES)), False),
    (3, 1, [ONES, 2*ONES, 3*ONES], [3*ONES], 3*ONES, True),
    (15, 3, [ONES, 2*ONES, 3*ONES, 4*ONES, 5*ONES, 6*ONES, 7*ONES, 8*ONES, 9*ONES, 10*ONES, 11*ONES,
             12*ONES, 13*ONES, 14*ONES, 15*ONES],
     [15*ONES, 14*ONES, 13*ONES], np.dstack((15*ONES, 14*ONES, 13*ONES)), True),
])
def test_step_images(steps: int, num_images: int, inputs: List[np.ndarray], actual: List[np.ndarray],
                     stack: np.ndarray, expected: bool) -> None:
    """
    Tests the step_images function.

    ASSUMES BLANK LIST WORKS PROPERLY.

    :param steps:      Number of steps forward in time
    :param num_images: Number of images to be used in the list
    :param inputs:     List of arrays that will be used as the input at each step
    :param actual:     The desired output list
    :param stack:      The desired output as a 128x128x(num_images*4) array
    :param expected:   The expected value of the comparison between the inputs and the actual/stack
    """
    image_list = blank_image_list(num_images)
    for i in range(steps):
        image_list = step_images(image_list, inputs[i])

    result = [(image_list[i] == actual[i]).all() for i in range(num_images)]
    result = False if False in result else True

    result_list = result == expected
    result_array = np.array_equal(np.dstack(image_list), stack) == expected

    print(f'The list result was: {result_list}\n'
          f'The array result was: {result_array}\n'
          f'The expected output for both was: {expected}')

    assert result_list and result_array


@pytest.fixture(scope="module")
def data_input() -> Tuple[List[Demo], List[Demo], str]:
    """
    This fixture is used for:
        - test_not_a_copy
        - test_saved_episode_lengths
        - test_low_dim_data_load_data
        - test_rgb_images_load_data
        - test_depth_images_load_data
        - test_mask_images_load_data
        - test_format_data
        - test_split_data

    It uses RLBench to generate two brand NEW episodes of a disl_pick_up_blue_cup task and saves them
    to disk at DISL_End_to_End_Learning/utils/test_episodes.

    We use deepcopy here (and through the aforementioned tests) to preserve a copy of the 'live' demos returned
    by the task.get_demos method. This is considered the ground truth for the next tests.

    After calling save_episodes on the deepcopy of 'live,' we immediately load the data back from disk with
    the load_data function. THIS FIXTURE ASSUMES THAT LOAD_DATA WORKS PROPERLY. This assumption is verified
    in the test_saved_episode_lengths test.

    Using the fixture at 'module' scope ensures that all the tests using this fixture receive the same
    'live' and 'disk' data and do not need to generate or load new episodes on their own.

    :returns: Tuple with the 'live' demo data, the 'disk' demo data, and the path to the demos
    """
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=True)
    env.launch()
    task = env.get_task(DislPickUpBlueCup)
    live = task.get_demos(2, live_demos=True)  # -> List[List[Observation]]
    env.shutdown()

    save_dir = join(getcwd(), 'utils', 'test_episodes')

    # Cannot save the version of live that we want to compare against.
    # Part of the save process removes the images so we need to save a deep copy of the
    # data structure.
    copy_live = deepcopy(live)
    save_episodes(copy_live, save_dir, 0)

    disk = [load_data(join(save_dir, 'variation0', 'episodes'), 0, obs_config),
            load_data(join(save_dir, 'variation0', 'episodes'), 1, obs_config)]

    return live, disk, save_dir


def test_not_a_copy(data_input: Tuple[List[Demo], List[Demo], str]) -> None:
    """
    Tests the deepcopy function to ensure that the live and disk demos (and their
    contents) are different objects in memory.

    :param data_input: Information returned from the data_input test fixture
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)

    result_demos = []
    result_obs = []

    for i in range(len(live)):
        result_demos.append(live[i] is disk[i])

        for j in range(len(live[i])):
            result_obs.append(live[i][j] is disk[i][j])

    demos_different = True if True not in result_demos else False
    obs_different = True if True not in result_obs else False

    print(f'Demo objects are different: {demos_different} \n'
          f'Observation objects are different: {obs_different}')

    assert demos_different and obs_different


@pytest.mark.parametrize("demo_num", [
    0,
    1,
])
def test_saved_episode_lengths(data_input: Tuple[List[Demo], List[Demo], str], demo_num: int) -> None:
    """
    Tests the save episode function by ensuring that the files on disk for the mask, RGB, and depth
    for each camera exist and contain the right number of steps.

    :param data_input: Information returned from the data_input test fixture
    :param demo_num:   Which of the two demos to test against
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)
    num_eps = len(live[demo_num])

    check_dir = join(save_dir, 'variation0', 'episodes', f'episode{demo_num}')
    results = []
    num_disk = {}

    for possible_dir in listdir(check_dir):
        location = join(check_dir, possible_dir)
        if isdir(location):
            num_disk[possible_dir] = len(listdir(location))
            results.append(num_disk[possible_dir] == num_eps)

    results = [False] if len(results) != 12 else results
    result = False if False in results else True

    print(f'For {check_dir} expected {num_eps} episodes... resulting dir/number dictionary: {num_disk}')

    assert result


@pytest.mark.parametrize("demo_num", [
    0,
    1,
])
def test_low_dim_data_load_data(data_input: Tuple[List[Demo], List[Demo], str], demo_num: int) -> None:
    """
    Tests load data's ability to reconstruct the low dimensional data.

    :param data_input: Information returned from the data_input test fixture
    :param demo_num:   Which of the two demos to test against
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)

    results = []
    for i in range(len(live[demo_num])):
        results.append(np.array_equal(disk[demo_num][i].joint_velocities,
                                      live[demo_num][i].joint_velocities))
        results.append(np.array_equal(disk[demo_num][i].joint_positions,
                                      live[demo_num][i].joint_positions))
        results.append(np.array_equal(disk[demo_num][i].joint_forces,
                                      live[demo_num][i].joint_forces))
        results.append(np.array_equal(disk[demo_num][i].gripper_pose,
                                      live[demo_num][i].gripper_pose))
        results.append(np.array_equal(disk[demo_num][i].gripper_matrix,
                                      live[demo_num][i].gripper_matrix))
        results.append(np.array_equal(disk[demo_num][i].gripper_joint_positions,
                                      live[demo_num][i].gripper_joint_positions))
        results.append(np.array_equal(disk[demo_num][i].gripper_touch_forces,
                                      live[demo_num][i].gripper_touch_forces))
        results.append(np.array_equal(disk[demo_num][i].wrist_camera_matrix,
                                      live[demo_num][i].wrist_camera_matrix))
        results.append(np.array_equal(disk[demo_num][i].task_low_dim_state,
                                      live[demo_num][i].task_low_dim_state))

        # Gripper open is a float; all others are np.ndarray
        results.append(disk[demo_num][i].gripper_open == live[demo_num][i].gripper_open)

    result = False if False in results else True

    assert result


@pytest.mark.parametrize("demo_num", [
    0,
    1,
])
def test_rgb_images_load_data(data_input: Tuple[List[Demo], List[Demo], str], demo_num: int) -> None:
    """
    Tests load data's ability to reconstruct the RGB info from each camera. Because we are comparing floats,
    np.allclose is used instead of np.array_equal since the values may change as they are scaled up when save
    and scaled back down when compared.

    :param data_input: Information returned from the data_input test fixture
    :param demo_num:   Which of the two demos to test against
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)

    results = []
    for i in range(len(live[demo_num])):
        left = np.allclose(a=disk[demo_num][i].left_shoulder_rgb / 255.,
                           b=live[demo_num][i].left_shoulder_rgb)

        right = np.allclose(a=disk[demo_num][i].right_shoulder_rgb / 255.,
                            b=live[demo_num][i].right_shoulder_rgb)

        wrist = np.allclose(a=disk[demo_num][i].wrist_rgb / 255.,
                            b=live[demo_num][i].wrist_rgb)

        front = np.allclose(a=disk[demo_num][i].front_rgb / 255.,
                            b=live[demo_num][i].front_rgb)

        for dif in [left, right, wrist, front]:
            results.append(dif)

    result = False if False in results else True

    assert result


@pytest.mark.parametrize("demo_num", [
    0,
    1,
])
def test_depth_images_load_data(data_input: Tuple[List[Demo], List[Demo], str], demo_num: int) -> None:
    """
    Tests load data's ability to reconstruct the depth images from each camera. Because we are comparing floats,
    np.allclose is used instead of np.array_equal since the values may change slightly as they are converted to RGB and
    back when save and loaded.

    :param data_input: Information returned from the data_input test fixture
    :param demo_num:   Which of the two demos to test against
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)

    results = []
    for i in range(len(live[demo_num])):
        left = np.allclose(a=disk[demo_num][i].left_shoulder_depth,
                           b=live[demo_num][i].left_shoulder_depth)

        right = np.allclose(a=disk[demo_num][i].right_shoulder_depth,
                            b=live[demo_num][i].right_shoulder_depth)

        wrist = np.allclose(a=disk[demo_num][i].wrist_depth,
                            b=live[demo_num][i].wrist_depth)

        front = np.allclose(a=disk[demo_num][i].front_depth,
                            b=live[demo_num][i].front_depth)

        for dif in [left, right, wrist, front]:
            results.append(dif)

    result = False if False in results else True

    assert result


@pytest.mark.parametrize("demo_num", [
    0,
    1,
])
def test_mask_images_load_data(data_input: Tuple[List[Demo], List[Demo], str], demo_num: int) -> None:
    """
    Tests load data's ability to reconstruct the image masks from each camera.

    Masks are integers but the saving process requires scaling modifications for
    RGB so FLOOR DIVISION BY 255 is used to format the saved values before comparison.

    np.allclose is used instead of np.array_equal since the values may change slightly when comparing floats.

    :param data_input: Information returned from the data_input test fixture
    :param demo_num:   Which of the two demos to test against
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)

    results = []

    for i in range(len(live[demo_num])):
        left = np.allclose(a=disk[demo_num][i].left_shoulder_mask // 255.,
                           b=live[demo_num][i].left_shoulder_mask)

        right = np.allclose(a=disk[demo_num][i].right_shoulder_mask // 255.,
                            b=live[demo_num][i].right_shoulder_mask)

        wrist = np.allclose(a=disk[demo_num][i].wrist_mask // 255.,
                            b=live[demo_num][i].wrist_mask)

        front = np.allclose(a=disk[demo_num][i].front_mask // 255.,
                            b=live[demo_num][i].front_mask)

        for dif in [left, right, wrist, front]:
            results.append(dif)

    result = False if False in results else True

    assert result


# test format data
@pytest.mark.parametrize("demo_num, pov", [
    (0, 'wrist'),
    (0, ''),
    (0, 'None'),
    (0, ['wrist', 'front', 'left_shoulder', 'right_shoulder']),
    (1, 'front'),
    (1, 'wrist'),
    (1, ['front']),
    (1, ['wrist']),
    (1, ['wrist', 'front']),
    (1, ['front', 'wrist']),
    (1, ['wrist', 'front', 'garbage / invalid pov'])
])
def test_format_data(data_input: Tuple[List[Demo], List[Demo], str], demo_num: int,
                     pov: Union[str, List[str]]) -> None:
    """
    Tests how format data scales the RGB images and used scale_array_down on the joints.
    Like format data, only the images we want to use - listed in pov - are compared for
    formating.

    checks step by step that the values match those in the 'live' demos.

    np.allclose is used instead of np.array_equal since the values may change slightly when comparing floats.

    ASSUMES THAT LOAD DATA HAS PROPERLY RECONSTRUCTED THE SAVED DEMO(S).

    :param data_input: Information returned from the data_input test fixture
    :param demo_num:   Which of the two demos to test against
    :param pov:        String of which point of view to format; list may be used for multiple POV inputs
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)

    data = format_data(disk[demo_num],
                       pov=pov)

    if type(pov) == str:
        pov = [pov]

    front = True if 'front' in pov else False
    wrist = True if 'wrist' in pov else False
    left_shoulder = True if 'left_shoulder' in pov else False
    right_shoulder = True if 'right_shoulder' in pov else False

    print(f'front={front}, wrist={wrist}, left_shoulder={left_shoulder}, right_shoulder={right_shoulder}')

    results_images = []
    results_joints = []

    l = None
    r = None
    w = None
    f = None

    for i, obs in enumerate(data):
        if left_shoulder:
            l = np.allclose(a=obs.left_shoulder_rgb,
                            b=live[demo_num][i].left_shoulder_rgb)
            results_images.append(l)
        if right_shoulder:
            r = np.allclose(a=obs.right_shoulder_rgb,
                            b=live[demo_num][i].right_shoulder_rgb)
            results_images.append(r)
        if wrist:
            w = np.allclose(a=obs.wrist_rgb,
                            b=live[demo_num][i].wrist_rgb)
            results_images.append(w)
        if front:
            f = np.allclose(a=obs.front_rgb,
                            b=live[demo_num][i].front_rgb)
            results_images.append(f)

        j = np.allclose(a=obs.joint_positions,
                        b=scale_pose_down(live[demo_num][i].joint_positions))
        results_joints.append(j)

    same_length = len(data) == len(live[demo_num])
    same_images = False if False in results_images else True
    same_joints = False if False in results_joints else True

    print(f'same_images was {same_images} with {len(results_images)//len(data)} POV in {pov} represented.\n'
          f'same_joints was {same_joints}. \n'
          f'same_length was {same_length}.')

    assert same_images and same_joints and same_length


@pytest.mark.parametrize("demo_num, num_images, pov", [
    (0, 4, 'front'),
    (0, 4, 'front'),
    (1, 4, 'front'),
    (0, 1, 'wrist'),
    (0, 8, 'front'),
    (1, 3, 'wrist'),
])
def test_split_data(data_input: Tuple[List[Demo], List[Demo], str], demo_num: int,
                    num_images: int, pov: str) -> None:
    """
    Tests split data's ability to create the inputs and labels for the network. Checks that the values
    at each step match those from the 'live' demos

    np.allclose is used instead of np.array_equal since the values may change slightly when comparing floats.

    ASSUMES THAT LOAD DATA HAS PROPERLY RECONSTRUCTED THE SAVED DEMO(S). ('disk')
    ASSUMES THAT FORMAT DATA HAS PROPERLY ADJUSTED THE VALUES IN THE SAVED DEMO(S). ('disk')

    :param data_input: Information returned from the data_input test fixture
    :param demo_num:   Which of the two demos to test against
    """
    # Must use deep copy before each test to copy the same data
    # to each function but NOT modify the data between functions.
    live, disk, save_dir = deepcopy(data_input)

    ###########################################
    # Arrange some containers for the testing #
    ###########################################
    live_image_list = blank_image_list(num_images=num_images)
    pov_list = [pov]
    front = True if 'front' in pov_list else False
    wrist = True if 'wrist' in pov_list else False
    left_shoulder = True if 'left_shoulder' in pov_list else False
    right_shoulder = True if 'right_shoulder' in pov_list else False

    input_angle_results = []
    input_action_results = []
    input_images_results = []

    label_angle_results = []
    label_action_results = []
    label_target_results = []
    label_gripper_results = []

    image = None

    ###################################
    # Act to split the formatted data #
    ###################################
    data = format_data(disk[demo_num], pov=pov)
    inputs, labels = split_data(episode=data, num_images=num_images, pov=pov)

    train_angles = inputs[0]
    train_action = inputs[1]
    train_images = inputs[2]

    label_angles = labels[0]
    label_action = labels[1]
    label_target = labels[2]
    label_gripper = labels[3]

    for step in range(len(train_angles)):
        ########################################
        # For each step, arrange the live data #
        ########################################
        if left_shoulder:
            image = np.dstack((live[demo_num][step].left_shoulder_rgb,
                               live[demo_num][step].left_shoulder_depth))
        if right_shoulder:
            image = np.dstack((live[demo_num][step].right_shoulder_rgb,
                               live[demo_num][step].right_shoulder_depth))
        if wrist:
            image = np.dstack((live[demo_num][step].wrist_rgb,
                               live[demo_num][step].wrist_depth))
        if front:
            image = np.dstack((live[demo_num][step].front_rgb,
                               live[demo_num][step].front_depth))
        try:
            live_label_angles = live[demo_num][step + 1].joint_positions
            live_label_action = live[demo_num][step + 1].gripper_open
        except IndexError:
            print(f'Caught exception at step {step} of {len(label_angles)}')
            live_label_angles = live[demo_num][step].joint_positions
            live_label_action = live[demo_num][step].gripper_open
        live_label_target = live[demo_num][step].task_low_dim_state[0][:3]
        live_label_gripper = live[demo_num][step].task_low_dim_state[-1][:3]

        ###############################################################################
        # At each step assert that the values are what we expect from the live record #
        ###############################################################################
        live_image_list = step_images(image_list=live_image_list, new_image=image)
        input_action_results.append(train_action[step] == live[demo_num][step].gripper_open)
        input_images_results.append(np.allclose(a=train_images[step],
                                                b=np.dstack(live_image_list)))
        label_action_results.append(label_action[step] == live_label_action)
        label_target_results.append(np.allclose(a=label_target[step],
                                                b=live_label_target))
        label_gripper_results.append(np.allclose(a=label_gripper[step],
                                                 b=live_label_gripper))

        # Note that we must used .copy() here since the joint positions would otherwise be changed
        # for later in the test when compared to the labels.
        input_angle_results.append(np.allclose(a=train_angles[step],
                                               b=scale_pose_down(live[demo_num][step].joint_positions.copy())))
        label_angle_results.append(np.allclose(a=label_angles[step],
                                               b=scale_pose_down(live_label_angles.copy())))

    #################################################
    # The assertions made in the loop are condensed #
    #################################################
    correct_angle_inputs = False if False in input_angle_results else True
    correct_action_inputs = False if False in input_action_results else True
    correct_image_inputs = False if False in input_images_results else True

    correct_angle_label = False if False in label_angle_results else True
    correct_action_label = False if False in label_action_results else True
    correct_target_label = False if False in label_target_results else True
    correct_gripper_label = False if False in label_gripper_results else True

    correct_inputs = correct_angle_inputs and correct_action_inputs and correct_image_inputs
    correct_labels = correct_angle_label and correct_action_label and correct_target_label and correct_gripper_label

    same_length = len(train_angles) == len(live[demo_num])

    print(f'Correct input angles:  {correct_angle_inputs}\n'
          f'Correct input action:  {correct_action_inputs}\n'
          f'Correct input images:  {correct_image_inputs}\n'
          f'Correct label angles:  {correct_angle_label}\n'
          f'Correct label action:  {correct_action_label}\n'
          f'Correct label target:  {correct_target_label}\n'
          f'Correct label gripper: {correct_gripper_label}\n'
          f'Same length: {same_length} ({len(train_angles)} split and {len(live[demo_num])} live)')

    assert correct_inputs and correct_labels and same_length
