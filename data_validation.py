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
from utils import save_demos
from utils import format_data
from utils import get_order
from utils import load_data
from utils import split_data


if __name__ == '__main__':
    num_live_demos = 3
    demo_save_path = 'data/misc_data/data_validation'

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
    pos_max = np.zeros(8)
    dep_max = 0
    img_max = 0
    pos_min = np.ones(8)*float('inf')
    dep_min = float('inf')
    img_min = float('inf')
    for L in range(len(live_demos)):
        for M in range(len(live_demos[L])):
            if live_demos[L][M].front_rgb.max() > img_max:
                img_max = live_demos[L][M].front_rgb.max()
            elif live_demos[L][M].front_rgb.min() < img_min:
                img_min = live_demos[L][M].front_rgb.min()

            if live_demos[L][M].front_depth.max() > dep_max:
                dep_max = live_demos[L][M].front_depth.max()
            elif live_demos[L][M].front_depth.min() < dep_min:
                dep_min = live_demos[L][M].front_depth.min()

            for I in range(len(live_demos[L][M].joint_positions)):
                if live_demos[L][M].joint_positions[I] > pos_max[I]:
                    pos_max[I] = live_demos[L][M].joint_positions[I]
                elif live_demos[L][M].joint_positions[I] < pos_min[I]:
                    pos_min[I] = live_demos[L][M].joint_positions[I]

            if live_demos[L][M].gripper_open > pos_max[-1]:
                pos_max[-1] = live_demos[L][M].gripper_open
            if live_demos[L][M].gripper_open < pos_min[-1]:
                pos_min[-1] = live_demos[L][M].gripper_open

    print(f'Summary of data from the {num_live_demos} demos: \n'
          f'Max image value: {img_max}\n'
          f'Min image value: {img_min}\n'
          f'Max depth value: {dep_max}\n'
          f'Min depth value: {dep_min}\n'
          f'Max positions: {pos_max}\n'
          f'Min positions: {pos_min}\n')

    save_demos(live_demos.copy(), demo_save_path)

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

    replay = True
    action = np.hstack((load_demos[0][0].joint_positions, load_demos[0][0].gripper_open))
    while replay:
        for i in range(len(live_demos)):
            task.reset()
            time.sleep(0.5)
            for j in range(len(live_demos[i])):
                # action = np.hstack(([i][j].joint_positions, load_demos[i][j].gripper_open))
                action[0] -= 0.1
                print(f'[{i}][{j}] Action is: {action}')
                current_obs = task.step(action)
                print(f'Current obs is {current_obs[0].joint_positions[0]}')
                # todo: seems to be a jolt between episodes, is there a way to avoid this?
        loop_again = input('Loop over the motions again? (y/n): ')
        if loop_again not in ['y', 'Y', 'yes', 'Yes']:
            replay = False


    print('Done!!')
    env.shutdown()
