from config import EndToEndConfig

from os.path import join
from os import listdir
from os import renames

import numpy as np
import time

if __name__ == '__main__':

    config = EndToEndConfig()

    config.list_data_set_directories()

    dir_num = None
    try:
        dir_num = int(input('\nSelect a directory # to check: '))
        if dir_num < 0 or dir_num > len(config.possible_data_set):
            exit('\n[Error] Please enter a valid index above zero.')
    except (IndexError, ValueError) as e:
        exit('\n[Error] Selection should be an integer. Exiting program.')

    broken_dataset_dir = join(config.data_root,
                              config.possible_data_set[dir_num],
                              'variation0',
                              'episodes')

    try:
        dir_to_rename = listdir(broken_dataset_dir)
        num_to_rename = len(dir_to_rename)
    except FileNotFoundError:
        dir_to_rename = None
        num_to_rename = None
        exit(f'[ERROR] The directory {broken_dataset_dir} was not found. Exiting program without '
             f'altering any files.')

    print(f'\n[Info] Renaming {num_to_rename} episodes in data set: {broken_dataset_dir}\n'
          f'[Info] Note, each episode is renumbered even if the data set is already continuous.')

    ans = input(f'\n[Info] Please verify that this is correct. Are you ready for the renaming to begin? (y/n): ')
    if ans not in ['y', 'yes', 'Y', 'Yes']:
        exit(f'[Warn] Answer: {ans} not recognized. Exiting program without altering any files.')

    print('[Info] Beginning to rename files, do not exit program...')

    count = []

    start_rename = time.perf_counter()

    # All files are given temp names first to avoid conflict
    for i, dir_name in enumerate(dir_to_rename):
        old_name = f'{broken_dataset_dir}/{dir_name}'
        count.append(len(listdir(f'{old_name}/front_rgb')))
        temp_name = f'{broken_dataset_dir}/episode{i}__temp'
        renames(old_name, temp_name)

    count = np.asarray(count)
    avg = np.mean(count)
    std = np.std(count)
    factor = 2
    fewer_name = []
    fewer_size = []
    more_name = []
    more_size = []
    min_size = [float('inf'), '']
    max_size = [-1, '']

    for i, dir_name in enumerate(listdir(broken_dataset_dir)):
        old_name = f'{broken_dataset_dir}/{dir_name}'
        size = len(listdir(f'{old_name}/front_rgb'))
        new_name = f'{broken_dataset_dir}/episode{i}'
        renames(old_name, new_name)

        if size < min_size[0]:
            min_size[0] = size
            min_size[1] = new_name
        elif size > max_size[0]:
            max_size[0] = size
            max_size[1] = new_name

        if size < (avg - factor*std):
            fewer_name.append(new_name)
            fewer_size.append(size)
        elif size > (avg + factor*std):
            more_name.append(new_name)
            more_size.append(size)

    end_rename = time.perf_counter()

    print(f'\n[Info] Finished renaming files.')

    s = '/'

    if len(fewer_name) != 0:
        print(f'\n[WARN] The following files seem to have fewer time steps in then than most. '
              f'All have at least {int(factor*std)} or fewer steps than the dataset average ({avg:.3f}). '
              f'It would be wise to manually check these')
        [print(f'{fewer_size[i]} at {possible_error.split(s)[-1]}') for i, possible_error in enumerate(fewer_name)]

    if len(more_name) != 0:
        print(f'\n[WARN] The following files seem to have more time steps in then than most. '
              f'All have at least {int(factor*std)} or more steps than the dataset average ({avg:.3f}). '
              f'It would be wise to manually check these')
        [print(f'{more_size[i]} at {possible_error.split(s)[-1]}') for i, possible_error in enumerate(more_name)]

    print(f'\n[Info] The average number of steps per episode was {avg:.3f} with standard deviation of {std:.3f}.')
    print(f'[Info] Max number of steps: {max_size[0]} at {max_size[1].split(s)[-1]}')
    print(f'[Info] Min number of steps: {min_size[0]} at {min_size[1].split(s)[-1]}')

    print(f'\n[Info] Successfully renamed {num_to_rename} files at: {broken_dataset_dir}. '
          f'Process took {end_rename - start_rename:.3f} seconds. Exiting program.')
