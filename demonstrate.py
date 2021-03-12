from tensorflow.keras.models import load_model
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from utils.utils import scale_pose
from config import EndToEndConfig
import numpy as np


if __name__ == '__main__':
    print('[Info] Starting demonstrate.py')

    config = EndToEndConfig()

    network_dir, network_name = config.get_trained_network()
    network = load_model(network_dir)
    print(f'\n[Info] Finished loading the network, {network_name}.')

    parsed_network_name = network_name.split('_')

    if 'pv4' in parsed_network_name or 'rnn-pv4' in parsed_network_name:
        print(f'\n[Info] Detected that the network uses 4 images. '
              f'Will structure inputs accordingly')
        four_deep = True
    else:
        four_deep = False

    pov = config.get_pov_from_name(parsed_network_name)

    task_name, imitation_task = config.get_task_from_name(parsed_network_name)

    num_demonstrations = int(input('\nEnter how many demonstrations to perform: '))
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

    evaluation_steps = num_demonstrations * demonstration_episode_length

    blank_image = np.zeros((128, 128, 4))
    image_t0 = blank_image.copy()
    image_t1 = blank_image.copy()
    image_t2 = blank_image.copy()
    image_t3 = blank_image.copy()

    obs = None
    for i in range(evaluation_steps):
        if i % demonstration_episode_length == 0:  # e.g. we're starting a new demonstration
            descriptions, obs = task.reset()
            print(f"[Info] Task reset: on episode {int(1+(i/demonstration_episode_length))} "
                  f"of {int(evaluation_steps / demonstration_episode_length)}")

        if pov == "wrist":
            image = np.dstack((obs.wrist_rgb, obs.wrist_depth))
        elif pov == "front":
            image = np.dstack((obs.wrist_rgb, obs.wrist_depth))
        else:
            image = np.dstack((obs.wrist_rgb, obs.wrist_depth))

        if not four_deep:
            pass
        else:
            image_t3 = image_t2.copy()
            image_t2 = image_t1.copy()
            image_t1 = image_t0.copy()
            image_t0 = image.copy()
            image = np.dstack((image_t0,
                               image_t1,
                               image_t2,
                               image_t3,))

        image = np.expand_dims(image, 0)
        state = np.expand_dims(np.append(obs.joint_positions, obs.gripper_open), 0)

        pred = network.predict(x=[state, image]).flatten()

        joint_pose = scale_pose(pred[:-1])
        gripper = pred[-1]
        action = np.append(joint_pose, gripper)

        obs, reward, terminate = task.step(action)

    env.shutdown()
    print(f'[Info] Successfully exiting program.')
