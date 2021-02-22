from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import PickUpCup
from os import listdir
import time

# Custom writen imports
from custom import save_demos

"""------ USER VARIABLES -----"""

# Specify requested task
requested_task = PickUpCup

# Select where the demos are saved here
root_save_path = 'dataset/misc_data/PickUpCup'
#    Note: from the root demos are saved .../variation#/episodes/episode#
#    the count for variations and the episodes within a variation start at 0
#    It is important that the dataset's collection of episodes be continuous
#    be careful of this when generating new demonstration episodes.
#
#    If demos already exist in the folder, the program will try to add more to
#    it until there are at least the desired number.

# Define the total number of demos you'd like in the folder
total_num_demos = 40

# Define how many demos to get in each loop.  Demos are saved after each loop.
demo_per_loop = 1

# Set to true to save resources be not displaying CoppeliaSim
headless = True

"""----- SET UP ENVIRONMENT -----"""
live_demos = True
DATASET = ''
obs_config = ObservationConfig()
obs_config.set_all(True)
action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
env = Environment(action_mode,
                  DATASET,
                  obs_config,
                  headless)
env.launch()

task = env.get_task(requested_task)

"""----- DEMO COLLECTION LOOP -----"""

full_save_path = root_save_path + '/variation0/episodes'
error_count = 0
max_consecutive_error = 20
prior_error = False

try:
    num_existing_demos = len(listdir(full_save_path))
except FileNotFoundError:
    num_existing_demos = 0

start_time = time.perf_counter()

while num_existing_demos < total_num_demos:
    begin_save_at = num_existing_demos

    try:
        #print('[Info] Beginning to collect demos for task.')
        demos = task.get_demos(demo_per_loop,
                               live_demos=live_demos)  # -> List[List[Observation]]

        #print('[Info] Beginning to save demos.')
        save_demos(demos,
                   root_save_path,
                   begin_save_at)

        #print('[Info] All demos for this loop were saved.')

        if prior_error:
            prior_error = False
    except RuntimeError:
        print('[Warn] Ran into a RuntimeError. Demos from this step were aborted.'
              ' Demo collection loop will continue.')
        if prior_error:
            error_count += 1
        prior_error = True

    if error_count >= max_consecutive_error:
        print(f'[Warn] experienced {error_count} consecutive errors. Exiting loop.')
        break

    try:
        num_existing_demos = len(listdir(full_save_path))
    except FileNotFoundError:
        num_existing_demos = 0

end_time = time.perf_counter()
delta_min = (end_time - start_time) / 60

total_num_steps = 0
for folder in listdir(full_save_path):
    total_num_steps += len(listdir(full_save_path + "/" + folder + '/front_rgb'))


print(f'[Info] Exiting program.\nA total of {len(listdir(full_save_path))} demos have been generated with {total_num_steps} '
      f'total data points.\nThe process took {delta_min:.3f} minutes. Files located in in {full_save_path}')
env.shutdown()


