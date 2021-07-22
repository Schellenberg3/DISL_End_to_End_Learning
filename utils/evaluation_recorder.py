from rlbench.environment import Demo
from rlbench.backend.observation import Observation

from copy import deepcopy

from typing import Union

from os.path import isdir
from os.path import join
from os.path import isfile
from os import mkdir

import numpy as np


class EvaluationRecorder(object):
    """Container for ground truth, prediction, and observed values"""
    def __init__(self, episode: Demo, obs: Observation, mode: str):
        self._mode = mode

        self._truth = episode

        self._header = ['Joint Position 1', 'Joint Position 2', 'Joint Position 3', 'Label Position 4',
                        'Joint Position 5', 'Joint Position 6', 'Joint Position 7',
                        'Joint Velocity 1', 'Joint Velocity 2', 'Joint Velocity 3', 'Label Velocity 4',
                        'Joint Velocity 5', 'Joint Velocity 6', 'Joint Velocity 7',
                        'Gripper Open',
                        'Target Position X', 'Target Position Y', 'Target Position Z',
                        'Gripper Position X', 'Gripper Position Y', 'Gripper Position Z',
                        f'Predicted {mode} 1', f'Predicted {mode} 2', f'Predicted {mode} 3', f'Predicted {mode} 4',
                        f'Predicted {mode} 5', f'Predicted {mode} 6', f'Predicted {mode} 7',
                        'Predicted Action',
                        'Predicted Target Position X', 'Predicted Target Position Y', 'Predicted Target Position Z',
                        'Predicted Gripper Position X', 'Predicted Gripper Position Y', 'Predicted Gripper Position Z',
                        'Obs Joint Position 1', 'Obs Joint Position 2', 'Obs Joint Position 3', 'Obs Label Position 4',
                        'Obs Joint Position 5', 'Obs Joint Position 6', 'Obs Joint Position 7',
                        'Obs Joint Velocity 1', 'Obs Joint Velocity 2', 'Obs Joint Velocity 3', 'Obs Label Velocity 4',
                        'Obs Joint Velocity 5', 'Obs Joint Velocity 6', 'Obs Joint Velocity 7',
                        'Obs Gripper Open',
                        'Obs Target Position X', 'Obs Target Position Y', 'Obs Target Position Z',
                        'Obs Gripper Position X', 'Obs Gripper Position Y', 'Obs Gripper Position Z',
                        ]
        initial = []
        # Get the ground truth
        initial += episode[0].joint_positions.tolist()
        initial += episode[0].joint_velocities.tolist()
        initial += [episode[0].gripper_open]
        initial += episode[0].task_low_dim_state[0][:3].tolist()
        initial += episode[0].task_low_dim_state[1][:3].tolist()
        # Buffer the predicted values with the initial observation
        initial += getattr(obs, f'joint_{mode}').tolist()
        initial += [obs.gripper_open]
        initial += obs.task_low_dim_state[0][:3].tolist()
        initial += obs.task_low_dim_state[1][:3].tolist()
        # Record the observations
        initial += obs.joint_positions.tolist()
        initial += obs.joint_velocities.tolist()
        initial += [obs.gripper_open]
        initial += obs.task_low_dim_state[0][:3].tolist()
        initial += obs.task_low_dim_state[1][:3].tolist()

        self.evaluation = [deepcopy(initial)]

        self._last_index = len(episode) - 1

        self._step = 0

    def update(self, prediction, obs: Observation):
        self._step += 1

        data = []

        # Save the ground truths
        s = self._step if self._step < self._last_index else self._last_index
        data += self._truth[s].joint_positions.tolist()
        data += self._truth[s].joint_velocities.tolist()
        data += [self._truth[s].gripper_open]
        data += self._truth[s].task_low_dim_state[0][:3].tolist()
        data += self._truth[s].task_low_dim_state[1][:3].tolist()
        # Save the predictions
        data += prediction[0].tolist()
        data += [prediction[1]]
        data += prediction[2].tolist()
        data += prediction[3].tolist()
        # Save the observations
        data += obs.joint_positions.tolist()
        data += obs.joint_velocities.tolist()
        data += [obs.gripper_open]
        data += obs.task_low_dim_state[0][:3].tolist()
        data += obs.task_low_dim_state[1][:3].tolist()

        self.evaluation.append(deepcopy(data))

    def save_eval(self, network_dir: str, dataset_name: str, ep: Union[int, str]):
        if not isdir(join(network_dir, 'dataset_evaluations')):
            mkdir(join(network_dir, 'dataset_evaluations'))

        save_as = join(network_dir, 'dataset_evaluations', f'{dataset_name}_episode_{ep}.csv')

        i = 0
        while isfile(save_as):
            save_as = save_as = join(network_dir, 'dataset_evaluations', f'{dataset_name}_episode_{ep}-{i}.csv')
            i += 1

        np.savetxt(save_as,
                   np.array(self.evaluation),
                   delimiter=",",
                   header=', '.join(self._header))
