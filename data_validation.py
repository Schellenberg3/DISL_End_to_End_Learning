# This file visually generates some number of demonstrations, saves them, reloads them, and compares
# Its purpose is to verify that the tools for saving, loading, and playing actions function properly

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import DislPickUpBlueCup
from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
import numpy as np
import time
# Custom writen imports
from disl_utils import save_demos
from disl_utils import format_data
from disl_utils import get_order
from disl_utils import load_data
from disl_utils import split_data


if __name__ == '__main__':
    num_live_demos = 3
    demo_save_path = 'datasets/misc_data/data_validation'

    domain_rand = False

    robot = 'panda'  # not sure if this will work for robots other than Panda

    obs_config = ObservationConfig()
    # obs_config.set_all(True)
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)

    if domain_rand:
        # todo: update DomainRand with robot to pass to env OR see what the options do for Environment
        rand_config = VisualRandomizationConfig(image_directory='../tests/unit/assets/textures')
        env = DomainRandomizationEnvironment(action_mode=action_mode,
                                             obs_config=obs_config,
                                             headless=False,
                                             randomize_every=RandomizeEvery.EPISODE,
                                             frequency=1,
                                             visual_randomization_config=rand_config,)
    else:
        env = Environment(action_mode=action_mode,
                          dataset_root='',
                          obs_config=obs_config,
                          headless=False,
                          robot_configuration=robot)

    env.launch()

    task = env.get_task(DislPickUpBlueCup)

    live_demos = task.get_demos(amount=num_live_demos,
                                live_demos=True)

    ''' 
    demos                  -> list[demonstration] contains info on each test demo
    demos[L]               -> list[list[observation]] list observations at each time step
    demos[L][M].front_rgb  -> list[list[class.member_variable]] float or int of state N at observation M, demo L
    '''

    demos = live_demos.copy()  # I don't like this either but it is needed because save_demos() sets images to None
    save_demos(demos, demo_save_path)
    del demos

    load_demos = []
    for i in range(num_live_demos):
        load_demos.append(load_data(demo_save_path + '/variation0/episodes', i, obs_config))

    steps = 0
    for i in range(len(live_demos)):
        for j in range(len(live_demos[i])):
            if live_demos[i][j].gripper_open != live_demos[i][j].gripper_open:
                print(f'[ERROR] Gripper position not saved properly at episode {i} step {j}')
            if (live_demos[i][j].joint_positions != live_demos[i][j].joint_positions).any():
                print(f'[ERROR] Joint position not saved properly at episode {i} step {j}')
            steps += 1

    print('live_demos == load_demos  :  ', live_demos == load_demos)

    replay = True
    while replay:
        for i in range(len(live_demos)):
            task.reset()
            time.sleep(0.5)
            for j in range(len(live_demos[i])):
                action = np.hstack((load_demos[i][j].joint_positions, load_demos[i][j].gripper_open))
                print(f'[{i}][{j}] Action is: {action}')
                task.step(action)
                # todo: seems to be a jolt between episodes, is there a way to avoid this?
        loop_again = input('Loop over the motions again? (y/n): ')
        if loop_again not in ['y', 'Y', 'yes', 'Yes']:
            replay = False


    print('Done!!')
    env.shutdown()
