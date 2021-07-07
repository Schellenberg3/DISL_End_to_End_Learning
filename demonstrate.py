from tensorflow.keras.models import load_model

from rlbench.action_modes import ArmActionMode
from rlbench.action_modes import ActionMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation

from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig

from utils.network_info import NetworkInfo
from utils.utils import scale_panda_pose
from utils.utils import blank_image_list
from utils.utils import step_images
from config import EndToEndConfig

from os.path import join
import numpy as np
import pickle

import matplotlib.pyplot as plt


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


def main():
    print('[Info] Starting demonstrate.py')

    config = EndToEndConfig()

    network_dir = config.get_trained_network()
    network = load_model(join(network_dir, network_dir.split('/')[-1] + '.h5'))

    pickle_location = join(network_dir, 'network_info.pickle')
    with open(pickle_location, 'rb') as handle:
        network_info: NetworkInfo = pickle.load(handle)

    print(f'\n[Info] Finished loading the network, {network_info.network_name}.')

    parsed_network_name = network_info.network_name.split('_')
    task_name, imitation_task = config.get_task_from_name(parsed_network_name)

    num_demonstrations = int(input('\nEnter how many demonstrations to perform (default 5): ') or 5)
    demonstration_episode_length = 40  # max steps per episode

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
    obs_config = ObservationConfig()

    env_type = 'regular'
    if env_type == 'random':
        rand_config = VisualRandomizationConfig(image_directory=config.domain_rand_textures)
        env = DomainRandomizationEnvironment(action_mode,
                                             obs_config=obs_config,
                                             headless=False,
                                             randomize_every=RandomizeEvery.EPISODE,
                                             frequency=1,
                                             visual_randomization_config=rand_config)
    else:
        env = Environment(action_mode=action_mode,
                          obs_config=obs_config,
                          headless=False,
                          robot_configuration='panda')

    env.launch()
    task = env.get_task(imitation_task)

    plt.ion()
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all')
    fig.suptitle('Error of Gripper and Target Estimates')
    fig.text(0.5, 0.04, 'Time Step', ha='center')
    fig.text(0.04, 0.5, 'Difference [m]', va='center', rotation='vertical')
    lines = []
    for ax, d in zip(axs, ['X', 'Y', 'Z']):
        ax.set_title(f'{d}-Direction')
        line_t, = ax.plot(0, 0, 'b-', label='Target')
        line_g, = ax.plot(0, 0, 'r-', label='Gripper')
        ax.plot(np.linspace(0, demonstration_episode_length, demonstration_episode_length),
                np.zeros(demonstration_episode_length), 'k-', label='Zero')
        lines.append((line_t, line_g))
    axs[0].legend(loc='lower left')

    evaluation_steps = num_demonstrations * demonstration_episode_length

    image_list = blank_image_list(network_info.num_images)

    obs = None
    for i in range(evaluation_steps):
        if i % demonstration_episode_length == 0:  # i.e. we're starting a new demonstration
            step = 0
            target_error = []
            gripper_error = []
            steps = []

            descriptions, obs = task.reset()
            image_list = blank_image_list(network_info.num_images)
            print(f"[Info] Task reset: on episode {int(1+(i/demonstration_episode_length))} "
                  f"of {int(evaluation_steps / demonstration_episode_length)}")
            input('Press enter to continue...')

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

        try:
            target_actual = task._task.cup.get_pose()
        except NameError:
            print('[Error] Unable to find target. Returning infinity as position')
            target_actual = np.array([np.inf, np.inf, np.inf])
        gripper_actual = obs.gripper_pose

        #################
        # Create Graphs #
        #################
        target_error.append(target_estimation - target_actual[:3])
        gripper_error.append(gripper_estimation - gripper_actual[:3])

        step += 1
        steps.append(step)

        t_e = np.array(target_error)
        g_e = np.array(gripper_error)

        for s, ax in enumerate(axs):
            line_t, line_g = lines[s]
            line_t.set_data(steps, t_e[:, s])
            line_g.set_data(steps, g_e[:, s])

            fig.canvas.draw()
            fig.canvas.flush_events()
            # print(f'steps={steps}\nt_e={t_e[:, s]}')
            ax.set_xlim(left=0, right=step)
            ax.set_ylim(bottom=min(min(g_e[:, s]), min(t_e[:, s])), top=max(max(g_e[:, s]), max(t_e[:, s])))

        #######################################################
        # Create action input and step the simulation forward #
        #######################################################
        action = np.append(joint_action, gripper_action)
        obs, reward, terminate = task.step(action)

    input('Press enter to exit...')
    env.shutdown()
    print(f'[Info] Successfully exiting program.')


if __name__ == '__main__':
    main()
