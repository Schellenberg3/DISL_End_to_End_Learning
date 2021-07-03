from config import EndToEndConfig

from os.path import join
from os.path import isdir
from os import listdir
from os import renames

import numpy as np
import time
import glob

if __name__ == '__main__':

    config = EndToEndConfig()
    broken_dataset_dir = config.get_data_set_directory('Select a directory # to check: ')

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
        temp_name = f'{broken_dataset_dir}/{dir_name}_tmp'
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
    broken_dataset = []

    missing_pkl = []

    total_size = 0

    for i, dir_name in enumerate(listdir(broken_dataset_dir)):
        old_name = f'{broken_dataset_dir}/{dir_name}'
        size = len(listdir(f'{old_name}/front_rgb'))
        new_name = f'{broken_dataset_dir}/episode{i}'
        renames(old_name, new_name)
        total_size += size

        pkl_count = len(glob.glob(join(new_name, 'low_dim_obs.pkl')))
        if pkl_count != 1:
            missing_pkl.append(new_name.split('/')[-1])

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

        sub_dirs = listdir(new_name)
        if len(sub_dirs) < 13:
            broken_dataset.append(new_name.split('/')[-1])
        else:
            for folder in sub_dirs:
                test_dir = join(new_name, folder)
                if isdir(test_dir) and len(listdir(test_dir)) != size:
                    broken_dataset.append(new_name.split('/')[-1])

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

    if len(missing_pkl) > 0:
        print(f'\n[WARN] Missing Pickle Files in {len(missing_pkl)} episode(s)! Check the following...')
        [print(f'{ep}') for ep in missing_pkl]
    else:
        print(f"\n[Info] Checked all episodes for 'low_dim_obs.pkl' files and found no missing data.")

    if len(broken_dataset) > 0:
        print(f'\n[WARN] Broken datasets in {len(missing_pkl)} episode(s)! Check the following...')
        [print(f'{ep}') for ep in broken_dataset]
    else:
        print(f"\n[Info] Checked all episodes for broken datasets and found no missing data.")   

    print(f'\n[Info] The dataset contains {total_size} datapoints across all {num_to_rename} episodes.')

    print(f'\n[Info] Successfully renamed {num_to_rename} files at: {broken_dataset_dir}. '
          f'Process took {end_rename - start_rename:.3f} seconds. Exiting program.')
