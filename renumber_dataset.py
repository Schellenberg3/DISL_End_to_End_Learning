import os
import time
import numpy as np

if __name__ == '__main__':
    broken_dataset_dir = 'datasets/randomized/training/DislPickUpBlueCup/variation0/episodes'

    try:
        dir_to_rename = os.listdir(broken_dataset_dir)
        num_to_rename = len(dir_to_rename)
    except FileNotFoundError:
        dir_to_rename = None
        num_to_rename = None
        exit(f'[ERROR] The directory {broken_dataset_dir} was not found. Exiting program without '
             f'altering any files.')

    print(f'[Info] Renaming {num_to_rename} episodes in a broken dataset at: {broken_dataset_dir}\n'
          f'Note that the original order of the numbers is not necessarily preserved')

    ans = input(f'\n[Info] Please verify that this is correct. Are you ready for the renaming to begin? (y/n): ')
    if ans not in ['y', 'yes', 'Y', 'Yes']:
        exit(f'[Warn] Answer: {ans} not recognized. Exiting program without altering any files.')

    count = []

    start_rename = time.perf_counter()

    # All files are given temp names first to avoid conflict
    for i, dir_name in enumerate(dir_to_rename):
        old_name = f'{broken_dataset_dir}/{dir_name}'
        count.append(len(os.listdir(f'{old_name}/front_rgb')))
        temp_name = f'{broken_dataset_dir}/episode{i}_temp'
        os.renames(old_name, temp_name)

    count = np.asarray(count)
    avg = np.mean(count)
    std = np.std(count)
    factor = 1.25
    possible_errors = []

    for i, dir_name in enumerate(os.listdir(broken_dataset_dir)):
        old_name = f'{broken_dataset_dir}/{dir_name}'
        size = len(os.listdir(f'{old_name}/front_rgb'))
        new_name = f'{broken_dataset_dir}/episode{i}'
        os.renames(old_name, new_name)
        if size < (avg - factor*std):
            possible_errors.append(new_name)

    end_rename = time.perf_counter()

    if len(possible_errors) != 0:
        print(f'\n[WARN] The following files seem to have fewer time steps in then than most '
              f'(the average was {avg} with standard deviation of +/-{std}). '
              f'It would be wise to manually check these')
        [print(possible_error) for possible_error in possible_errors]

    print(f'\n[Info] Successfully renamed {num_to_rename} files at: {broken_dataset_dir}. '
          f'Process took {end_rename - start_rename} seconds. Exiting program.')
