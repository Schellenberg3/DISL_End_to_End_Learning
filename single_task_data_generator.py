from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import DislPickUpBlueCup
from rlbench.tasks import PickUpCup
from disl_utils import save_demos
from multiprocessing import Process
from os import listdir
import os
import time
import pathlib


def multiprocess_demos(mp_action_mode,
                       mp_obs_config,
                       mp_headless,
                       mp_request,
                       mp_start_at,
                       mp_demos_per_loop,
                       mp_root_save_path,
                       mp_domain_rand,
                       mp_dr_images):

    if not mp_domain_rand:
        mp_env = Environment(action_mode=mp_action_mode,
                             obs_config=mp_obs_config,
                             headless=mp_headless)
    else:
        # Configuration textures borrowed from RLBench
        # todo: generate our own textures - different levels of textures?
        im_path = pathlib.Path().absolute() / mp_dr_images
        mp_rand_config = VisualRandomizationConfig(image_directory=im_path)
        # todo: does domain randomization environment ONLY work with panda?
        # todo: what does frequency do?
        mp_env = DomainRandomizationEnvironment(action_mode=mp_action_mode,
                                                obs_config=mp_obs_config,
                                                headless=mp_headless,
                                                randomize_every=RandomizeEvery.EPISODE,
                                                frequency=1,
                                                visual_randomization_config=mp_rand_config)

    mp_env.launch()

    mp_task = mp_env.get_task(requested_task)

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
            mp_demos = mp_task.get_demos(mp_demos_per_loop, live_demos=True)
            save_demos(mp_demos, mp_root_save_path, mp_begin_save_at)

            if mp_prior_error:
                mp_prior_error = False
                mp_error_count = 0
        except RuntimeError:
            if mp_prior_error:
                mp_error_count += 1
            mp_prior_error = True

        if mp_error_count >= mp_max_consecutive_error:
            print(f'[WARN] experienced {mp_error_count} consecutive RuntimeError with CopelliaSim.'
                  f'Abandoning the process. Dataset is likely broken near demonstration episode {mp_begin_save_at}.')
            break

        if not mp_prior_error:
            mp_remaining -= mp_demos_per_loop
            mp_generated += mp_demos_per_loop

    mp_env.shutdown()


if __name__ == '__main__':
    """------ USER VARIABLES -----"""
    # todo: make this code more general for variations of tasks
    # Specify requested task
    requested_task = DislPickUpBlueCup #PickUpCup

    # should the domain be randomized?
    domain_rand = True
    dr_images = 'datasets/assets/textures'  # relative path from this file's directory

    # Select where the demos are saved here
    root_save_path = 'datasets/randomized/training/DislPickUpBlueCup'
    #    Note: from the root demos are saved .../variation#/episodes/episode#
    #    the count for variations and the episodes within a variation start at 0
    #    It is important that the dataset's collection of episodes be continuous
    #    be careful of this when generating new demonstration episodes.
    #
    #    If demos already exist in the folder, the program will try to add more to
    #    it until there are at least the desired number.

    # Define the total number of demos you'd like in the folder
    num_total_demos = 2000

    # Define how many demos to get in each loop.  Demos are saved after each loop.
    num_demo_per_loop = 1

    # Set to true to save resources by not displaying CoppeliaSim
    headless = True

    """----- SET UP -----"""
    live_demos = True
    obs_config = ObservationConfig()
    obs_config.set_all(True)  # todo: Adjust this for UR5
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)

    processes = []
    mp_start_time = time.perf_counter()

    full_save_path = root_save_path + '/variation0/episodes'

    try:
        num_existing_demos = len(listdir(full_save_path))
    except FileNotFoundError:
        print(f'[Warn] It looks like {root_save_path} might not exist or is not set up properly. '
              f'When creating demos the directory {full_save_path} will be created.')
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


        print(f'[Info] A total of {len(listdir(full_save_path))} '
              f'demonstration episodes have been collected with {total_num_steps} total data points.'
              f'\n[Info] All demonstration episodes are located in: {full_save_path}'
              f'\n[Info] Exiting program successfully.')
        exit()

    demo_per_process = int(required_new_demos / os.cpu_count())
    demo_per_process_remainder = required_new_demos % os.cpu_count()

    print(f'[Info] Requested a total of {num_total_demos} demonstration episodes and '
          f'found {num_existing_demos} at the desired location. '
          f'\n[Info] Will generate {required_new_demos} new demonstration episodes at {full_save_path}')

    ans = input(f'\n[Info] Are you ready for the task collection to begin? (y/n): ')
    if ans not in ['y', 'yes', 'Y', 'Yes']:
        exit(f'[Warn] Answer: {ans} not recognized. Exiting program without generating demonstrations.')

    num_start_at = num_existing_demos
    for i in range(os.cpu_count()):

        if i < (os.cpu_count() - 1) and demo_per_process_remainder > 0:
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

    # todo: add exit cond
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
