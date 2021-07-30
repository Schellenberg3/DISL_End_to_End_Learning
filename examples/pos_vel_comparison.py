from config import EndToEndConfig
from utils.utils import load_data
from utils.utils import scale_panda_pose
from os import listdir
import matplotlib.pyplot as plt
import numpy as np


def main():
    config = EndToEndConfig()
    dataset_dir = config.get_data_set_directory()
    ep = int(input(f'Enter what episode to use (0-{len(listdir(dataset_dir))-1} available. default is 0): ') or 0)
    episode = load_data(dataset_dir, ep, config.rlbench_obsconfig)

    pos = np.array([scale_panda_pose(step.joint_positions, direction='down') for step in episode])
    vel = np.array([step.joint_velocities for step in episode])

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all')
    fig.suptitle('Joint position and joint Velocity')
    joint_numbers = ['1', '2', '3', '4', '5', '6', '7']
    labels = ['angle (scaled)', 'velocity']
    k = 0
    for j, (label, data) in enumerate(zip(labels, [pos, vel])):
        for i, joint in enumerate(joint_numbers):
            try: 
                axs[j].set_title(f'joint {labels[j]}')
                axs[j].plot(data[:, i])
            except IndexError:
                k += 1
                print(f'Error {k}')
    plt.show()


if __name__ == '__main__':
    main()

