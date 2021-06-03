from rlbench.action_modes import ArmActionMode
from rlbench.action_modes import ActionMode

from rlbench.environment import Environment

from rlbench.observation_config import ObservationConfig

from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig

from utils.utils import save_episodes
from utils.utils import check_yes
from config import EndToEndConfig

from multiprocessing import Process
from os.path import join
from os import cpu_count
from os import listdir

from typing import Union

import pathlib
import time


def get_env(rand: bool, act_mode: ActionMode, obs: ObservationConfig, image_dir: str,
            head: bool = True) -> Union[Environment, DomainRandomizationEnvironment]:
    if not rand:
        env = Environment(action_mode=act_mode, obs_config=obs, headless=head)
    else:
        im_path = pathlib.Path().absolute() / image_dir
        mp_rand_config = VisualRandomizationConfig(image_directory=im_path)
        # todo: Domain randomization code forces panda. Why? Can this be changed?
        env = DomainRandomizationEnvironment(action_mode=act_mode,
                                             obs_config=obs,
                                             headless=head,
                                             visual_randomization_config=mp_rand_config,
                                             randomize_every=RandomizeEvery.EPISODE)
    return env


def multiprocess_demos(mp_action_mode,
                       mp_obs_config,
                       mp_headless,
                       mp_request,
                       mp_start_at,
                       mp_demos_per_loop,
                       mp_root_save_path,
                       mp_domain_rand,
                       mp_dr_images):
    mp_env = get_env(rand=mp_domain_rand, act_mode=mp_action_mode,
                     obs=mp_obs_config, image_dir=mp_dr_images, head=mp_headless)
    mp_env.launch()

    mp_error_count = 0
    mp_max_consecutive_error = 30

    mp_generated = 0
    mp_prior_error = False

    mp_remaining = mp_request

    while mp_remaining > 0:
        mp_begin_save_at = mp_start_at + mp_generated

        if mp_remaining < mp_demos_per_loop:
            mp_demos_per_loop = mp_remaining

        try:
            mp_task = mp_env.get_task(requested_task)
            mp_demos = mp_task.get_demos(mp_demos_per_loop, live_demos=True)
            save_episodes(mp_demos, mp_root_save_path, mp_begin_save_at)
            del mp_task

            if mp_prior_error:
                mp_prior_error = False
                mp_error_count = 0
        except RuntimeError:
            if mp_prior_error:
                mp_error_count += 1
            mp_prior_error = True

        if mp_error_count >= mp_max_consecutive_error:
            print(f'[WARN] experienced {mp_error_count} consecutive RuntimeError with CopelliaSim. '
                  f'Abandoning the process. Dataset is likely broken near demonstration episode {mp_begin_save_at}.')
            break

        if not mp_prior_error:
            mp_remaining -= mp_demos_per_loop
            mp_generated += mp_demos_per_loop

    mp_env.shutdown()


if __name__ == '__main__':
    print('[Info] Starting generate_episodes.py')

    config = EndToEndConfig()

    task_name, requested_task = config.get_task_from_user()

    # Define the total number of demos you'd like in the folder
    num_total_demos = int(input('\nHow many episodes should be generated? '))

    tag = input('\nWhat tag should the directory have? Testing (default), training, misc:  ').lower()
    if tag not in ['testing', 'training', 'misc']:
        tag = 'testing'

    if check_yes('\nWill the scene be randomized? (y/n) '):
        domain_rand = True
        tag += '_randomized'
    else:
        domain_rand = False

    dr_images = config.domain_rand_textures

    root_save_path = join(config.data_root,
                          tag,
                          task_name)
    full_save_path = join(root_save_path,
                          'variation0',
                          'episodes')
    #    Note: from the root demos are saved .../variation#/episodes/episode#
    #    the count for variations and the episodes within a variation start at 0
    #    It is important that the dataset's collection of episodes be continuous
    #    be careful of this when generating new demonstration episodes.
    #
    #    If demos already exist in the folder, the program will try to add more to
    #    it until there are at least the desired number. It will match the existing
    #    numbering too.  check_episodes.py exists to resolve any errors if this fails.

    num_demo_per_loop = 1

    headless = True  # To save resources by not displaying CoppeliaSim
    live_demos = True

    obs_config = ObservationConfig()
    obs_config.task_low_dim_state = True
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)

    processes = []
    mp_start_time = time.perf_counter()

    try:
        num_existing_demos = len(listdir(full_save_path))
    except FileNotFoundError:
        print(f'[Warn] It looks like {root_save_path} might not exist or is not set up properly. '
              f'Before generating episodes the directory {full_save_path} will be created.')
        num_existing_demos = 0

    required_new_demos = num_total_demos - num_existing_demos

    if required_new_demos <= 0:
        print(f'[ERROR] Invalid number of demos requested. Providing summary of dataset instead.')

        total_num_steps = 0
        try:
            for folder in listdir(full_save_path):
                total_num_steps += len(listdir(full_save_path + "/" + folder + '/front_rgb'))
        except FileNotFoundError:
            print(f'[ERROR] No dataset exists at {full_save_path} or the one that does is broken.')
            exit()

        print(f'\n[Info] A total of {len(listdir(full_save_path))} '
              f'demonstration episodes have been collected with {total_num_steps} total data points.'
              f'\n[Info] All demonstration episodes are located in: {full_save_path}'
              f'\n[Info] Exiting program successfully.')
        exit()

    demo_per_process = int(required_new_demos / cpu_count())
    demo_per_process_remainder = required_new_demos % cpu_count()

    print(f'[Info] Requested a total of {num_total_demos} demonstration episodes and '
          f'found {num_existing_demos} at the desired location. '
          f'\n[Info] Will generate {required_new_demos} new demonstration episodes at {full_save_path}')

    ans = input(f'\n[Info] Are you ready for the episode generation to begin? (y/n): ')
    if ans not in ['y', 'yes', 'Y', 'Yes']:
        exit(f'[Warn] Answer: {ans} not recognized. Exiting program without generating demonstrations.')

    num_start_at = num_existing_demos
    for i in range(cpu_count()):

        if i < (cpu_count() - 1) and demo_per_process_remainder > 0:
            num_request = demo_per_process + 1
            demo_per_process_remainder -= 1
        else:
            num_demo_per_loop = 1
            num_request = demo_per_process

        processes.append(Process(target=multiprocess_demos,
                                 args=(action_mode,
                                       obs_config,
                                       headless,
                                       num_request,
                                       num_start_at,
                                       num_demo_per_loop,
                                       root_save_path,
                                       domain_rand,
                                       dr_images)
                                 )
                         )

        print(f'[Info] Registered process {i} to generate {num_request} demos. '
              f'Demos will be saved as episodes {num_start_at} to {num_start_at+num_request-1}')

        num_start_at += num_request

    [process.start() for process in processes]

    print(f'[Info] All {len(processes)} processes started. Generating demonstrations...')

    [process.join() for process in processes]

    mp_end_time = time.perf_counter()

    print(f'[Info] All {len(processes)} processes rejoined main.')

    delta_min = (mp_end_time - mp_start_time) / 60

    total_num_steps = 0
    for folder in listdir(full_save_path):
        total_num_steps += len(listdir(full_save_path + "/" + folder + '/front_rgb'))

    num_after_gen = len(listdir(full_save_path))
    if num_total_demos < num_after_gen:
        print(f'[ERROR] After generation only {num_after_gen} demonstration episodes exist. '
              f'The dataset is broken but potentially fixable with renumber_dataset.py')

    print(f'[Info] A total of {num_after_gen} '
          f'demonstration episodes have been collected with {total_num_steps} total data points.'
          f'\n[Info] The process of adding {required_new_demos} new demonstration episodes '
          f'took {delta_min:.3f} minutes. '
          f'\n[Info] All demonstration episodes are located in: {full_save_path}'
          f'\n[Info] Exiting program successfully.')
