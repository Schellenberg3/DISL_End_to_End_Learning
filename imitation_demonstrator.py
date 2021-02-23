from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import DislPickUpBlueCup
from rlbench.tasks import PickUpCup
from rlbench.tasks import ReachTarget
from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
import os
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


if __name__ == '__main__':
    """------ USER VARIABLES -----"""

    model_dir = f'imitation_trained'

    env_type = 'regular'  # or random

    imitation_task = DislPickUpBlueCup

    num_demonstrations = 5
    demonstration_episode_length = 120  # max steps per episode

    # todo: handle using ur5 and non-panda robots
    robot = 'panda'  # Only a naive change.  Will cause errors if set to

    """------ SET UP -----"""

    try:
        print(f'[Info] Searching for models...\n\nModel #  :  MODEL_DIRECTORY')
        for i, model_name in enumerate(os.listdir(model_dir)):
            print(f'Model {i}  :  {model_name}')

        selected_num = int(input(f'\nPlease select the Model # to demonstrate (e.g. enter 1, 2, or 3): '))
        selected_model = os.listdir(model_dir)[selected_num]
        model = tf.keras.models.load_model(model_dir + '/' + selected_model)

        print(f'[Info] Successfully loaded Model {selected_num}  :   {selected_model}')
    except (FileNotFoundError, TypeError, ValueError, IndexError) as e:
        model = None
        print(f'[ERROR] Exception "{e}" was raised. Exiting program. '
              f'Could not find models in {model_dir} or invalid model requested.')
        exit()

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    if env_type == 'random':
        rand_config = VisualRandomizationConfig(image_directory='../tests/unit/assets/textures')
        # todo: does domain randomization environment ONLY use panda?
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
                          robot_configuration=robot)

    env.launch()

    task = env.get_task(imitation_task)

    evaluation_steps = num_demonstrations * demonstration_episode_length

    obs = None
    for i in range(evaluation_steps):
        if i % demonstration_episode_length == 0:  # e.g. we're starting a new demonstration
            descriptions, obs = task.reset()
            print(f"[Info] Task reset: on episode {int(1+(i/demonstration_episode_length))} "
                  f"of {int(evaluation_steps / demonstration_episode_length)}")

        image = np.expand_dims(np.dstack((obs.front_rgb, obs.front_depth)), 0)
        state = np.expand_dims(np.append(obs.joint_positions, obs.gripper_open), 0)
        action = model.predict(x=[state, image])
        print(action[0][7])
        # todo: Why aren't we actually grasping the cup?
        #     Gripper value 1 -> open, gripper value 0 -> closed
        #     Seems like something isn't right in training
        #
        #     Should review the model's structure, training. Verify that is *is* acting as expected
        #     Write a module to verify the data's structure
        obs, reward, terminate = task.step(action.flatten())

    env.shutdown()
    print(f'[Info] Successfully exiting program.')
