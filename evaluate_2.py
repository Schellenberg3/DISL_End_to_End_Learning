from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session

from rlbench.action_modes import ArmActionMode
from rlbench.backend.observation import Observation

from utils.recorder import Recorder
from utils.utils import scale_panda_pose
from utils.utils import blank_image_list
from utils.utils import step_images
from utils.utils import get_order
from utils.utils import load_data
from config import EndToEndConfig

from typing import Union
from typing import Tuple
from typing import List

from os.path import join
from os import listdir
import numpy as np
import gc


def get_image(obs: Observation, pov: str) -> np.ndarray:
    """
    Gets depth image from an observation based on the point of view.

    :param obs: RLBench observation at a given time step
    :param pov: String of what point of view to return an image of

    :return: 128x128x4 depth image as a numpy array.
    """
    if pov == "wrist":
        image = np.dstack((obs.wrist_rgb, obs.wrist_depth))
    elif pov == "front":
        image = np.dstack((obs.wrist_rgb, obs.wrist_depth))
    else:
        image = np.dstack((obs.wrist_rgb, obs.wrist_depth))
    return image


def get_episode_info(config: EndToEndConfig) -> Tuple[Union[List[int], List[str]], str, bool]:
    """
    Gets a list of episode numbers to recreate.

    :param config: EndToEnd config object that holds the dataset information.

    :returns:  A tuple of: (1) A list of episode numbers as either integers or strings to be loaded,
               (2) file path for the dataset directory, and (3) a bool that is true if the dataset is randomized.
    """
    config.list_data_set_directories()
    dir_num = None
    try:
        dir_num = int(input('\nSelect a directory reproduce from: '))
        if dir_num < 0 or dir_num > len(config.possible_data_set):
            exit('\n[Error] Please enter a valid index above zero.')
    except (IndexError, ValueError) as e:
        exit('\n[Error] Selection should be an integer. Exiting program.')

    dataset_dir = join(config.data_root,
                       config.possible_data_set[dir_num],
                       'variation0',
                       'episodes')
    dir_len = len(listdir(dataset_dir))

    print(f"\nThere are {dir_len} episodes available. Enter a list of episodes or a number followed by "
          f"'!' to get that number of random episodes...")
    eps = input("Enter which episode(s) to load (default is '5!'): ") or '5!'

    randomized = True if 'randomized' in config.possible_data_set[dir_num].split('/')[0].split('_') else False

    if '!' in eps:
        eps = get_order(int(eps[:-1]), dir_len, 1)
    else:
        eps = eps.split()

    return eps, dataset_dir, randomized


def main():
    print('[Info] Starting demonstrate.py')

    config = EndToEndConfig()

    network_dir = config.get_trained_network_dir()
    network, network_info = config.load_trained_network(network_dir)

    print(f'\n[Info] Finished loading the network, {network_info.network_name}.')

    _, imitation_task = config.get_task_from_name(network_info.network_name.split('_'))

    eps, dataset_dir, randomized = get_episode_info(config)

    mode = 'velocities'
    config.set_action_mode(mode)

    env = config.get_env(randomized=randomized)
    env.launch()

    record = Recorder(network_dir=network_dir)

    demonstration_episode_length = 100  # max steps per episode
    for i, ep in enumerate(eps):
        task = env.get_task(imitation_task)

        episode = load_data(dataset_dir, ep, config.rlbench_obsconfig)
        episode.restore_state()

        descriptions, obs = task.reset()
        image_list = blank_image_list(network_info.num_images)

        print(f'\n[Info] Reproducing episode {ep}. On demo {i+1} of {len(eps)}')
        input('Press enter to continue...')
        for s in range(demonstration_episode_length):
            record.add_image(obs.front_rgb)
            ##############################################################
            # Collect prediction information from the latest observation #
            ##############################################################
            image = get_image(obs, network_info.pov)
            image_list = step_images(image_list, image)
            image_input = np.expand_dims(np.dstack(image_list), 0)

            gripper_input = np.expand_dims(obs.gripper_open, 0)

            joints_input = scale_panda_pose(obs.joint_positions, 'down')  # to [0, 1] for prediction
            joints_input = np.expand_dims(joints_input, 0)

            #######################
            # Make the prediction #
            #######################
            prediction = network.predict(x=[joints_input, gripper_input, image_input])

            ##########################################################
            # Parse prediction for the actions and auxiliary outputs #
            ##########################################################
            joint_action = prediction[0].flatten()
            if config.rlbench_actionmode.arm == ArmActionMode.ABS_JOINT_POSITION:
                joint_action = scale_panda_pose(joint_action, 'up')   # from [0, 1] to joint's proper values

            gripper_action = np.argmax(prediction[1].flatten())

            target_estimation = prediction[2].flatten()

            gripper_estimation = prediction[3].flatten()

            #####################
            # Get actual values #
            #####################
            try:
                joint_label = getattr(episode[s], f'joint_{mode}')
                action_label = episode[s].gripper_open
                target_label = episode[s].task_low_dim_state[0][:3]
                gripper_label = episode[s].task_low_dim_state[1][:3]
            except IndexError:
                print('missed label')
                joint_label = np.zeros_like(joint_action)
                action_label = np.zeros_like(gripper_action)
                target_label = np.zeros_like(target_estimation)
                gripper_label = np.zeros_like(gripper_estimation)


            #######################################################
            # Create action input and step the simulation forward #
            #######################################################
            action = np.append(joint_action, gripper_action)
            obs, reward, terminate = task.step(action)

        record.save_gif(f'{ep}')

        del network
        clear_session()
        gc.collect()
        network = load_model(join(network_dir, network_dir.split('/')[-1] + '.h5'))

    input('Press enter to exit...')
    env.shutdown()
    print(f'[Info] Successfully exiting program.')


if __name__ == '__main__':
    main()
