from rlbench.sim2real.domain_randomization import VisualRandomizationConfig
from rlbench.sim2real.domain_randomization_environment import DomainRandomizationEnvironment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes import ArmActionMode
from rlbench.action_modes import ActionMode
from rlbench.environment import Environment
from rlbench.tasks import DislPickUpBlueCup
from rlbench.tasks import ReachTarget
from rlbench import RandomizeEvery

from utils.training_info import TrainingInfo
from utils.network_info import NetworkInfo
from utils.networks import NetworkBuilder
from utils.utils import alpha_numeric_sort

from tensorflow.keras.models import load_model
from tensorflow.keras import Model

from os.path import dirname
from os.path import realpath
from os.path import join
from os import listdir

from typing import Tuple
from typing import Union
from typing import List
from typing import Any

import pickle


class EndToEndConfig:
    def __init__(self):
        """ Container for common variables and getter/setters used
        throughout the code.
        """
        # DO NOT MOVE THIS FILE FROM THE MAIN FOLDER. WILL BREAK DIRECTORY LOCATION ASSUMPTION
        self.data_root = join(dirname(realpath(__file__)), 'data')
        self.possible_data_set = []

        self.network_root = join(dirname(realpath(__file__)), 'networks')
        self.network_sub_dir = ['imitation', 'reinforcement']
        self._possible_network = []

        self.domain_rand_textures = join(dirname(realpath(__file__)), 'utils', 'textures')

        self.pov = ['front', 'wrist']

        # task name : RLBench object
        self.tasks = {"ReachTarget": ReachTarget,
                      "DislPickUpBlueCup": DislPickUpBlueCup,
                      }
        self.default_task = ["ReachTarget", ReachTarget]

        ###########################################################################################
        # We contain a few RLBench configurations within this configuration to ensure consistency #
        ###########################################################################################
        self.rlbench_obsconfig = ObservationConfig()
        self.rlbench_obsconfig.task_low_dim_state = True

        self.rlbench_actionmode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)

        # Important to keep the blacklist updated; no errors are thrown if the item does not exist in the scene
        self.rlbench_random_config = VisualRandomizationConfig(image_directory=self.domain_rand_textures,
                                                               blacklist=['cup_visual',
                                                                          'nonexistant_test',
                                                                          'Panda_link0_visual',
                                                                          'Panda_link1_visual',
                                                                          'Panda_link2_visual',
                                                                          'Panda_link3_visual',
                                                                          'Panda_link4_visual',
                                                                          'Panda_link5_visual',
                                                                          'Panda_link6_visual',
                                                                          'Panda_link7_visual',
                                                                          'Panda_gripper_visual',
                                                                          'Panda_leftfinger_visual',
                                                                          'Panda_rightfinger_visual'])

    def get_task_from_name(self, parsed_name: List[str]):
        """ Uses the network name or directory name to select the
        correct task to load into CoppeliaSim. If it cannot find the
        task, self.get_task_from_user() is called.

        :param parsed_name: Network name split
        :return: (Tasks Name, RLBench Task)
        """
        task_name = None
        task_class = None

        for parse in parsed_name:
            if parse in self.tasks.keys():
                task_name = parse
                task_class = self.tasks[parse]
                break

        if task_name:
            print(f'\n[Info] Automatically selected {task_name} as the RLBench task.')
            return task_name, task_class
        else:
            print(f'\n[Warn] Could not look up task. Please select this manually.')
            return self.get_task_from_user()

    def get_task_from_user(self) -> Tuple[str, Any]:
        """ Asks user to select from the list of tasks in the
        self.tasks dictionary. Returns the name and the selected
        class

        :return: (Tasks Name, RLBench Task)
        """

        print(f'\nPlease select one of the tasks to use: ')
        key_list = []
        for i, key in enumerate(self.tasks.keys()):
            print('{:.<10s}{:.<2s}'.format(f'Task {i}', f'{key}'))
            key_list.append(key)

        try:
            task_num = int(input('\nEnter # of desired task: '))

            if task_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            task_name = key_list[task_num]
            task_class = self.tasks[task_name]
            return task_name, task_class

        except (ValueError, IndexError) as e:
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    # todo: consider removing
    def get_pov_from_name(self, parsed_name: List[str]) -> str:
        """ Takes a parsed name (from a saved network) and looks for information
        on the networks trained point of view. Returns this or asks for user
        input if no info is found

        :param parsed_name: Network name split
        :return: Either "wrist" or "front"
        """
        if 'wrist' in parsed_name:
            print(f'\n[Info] Automatically selected "wrist" as the POV.')
            return 'wrist'
        elif 'front' in parsed_name:
            print(f'\n[Info] Automatically selected "front" as the POV.')
            return 'front'
        else:
            print(f'[Warn] Unable to infer networks point of view.')
            return self.get_pov_from_user()

    @staticmethod
    def get_pov_from_user() -> str:
        """ If point of view cannot be inferred from the network's name
        this prompts the user to select which camera view to use.

        :return: Either "wrist" or "front"
        """
        pov = input('Please enter what camera point of view to use, front (default) or wrist: ') or 'front'
        if pov in ['front', 'Front', 'wrist', 'Wrist']:
            pov = pov.lower()
            print(f'[Info] Using {pov} point of view.')
            return pov
        else:
            print(f'[Warn] Input "{pov}" is not a supported option. Defaulting to front.')
            return 'front'

    @staticmethod
    def get_task_name(name: str):
        split = name.split('_')
        if len(split) == 1:
            return split[0], False
        elif 'randomized' in split:
            return split[0], True
        else:
            raise Exception(f'Could not parse training directory name: {name}')

    def get_env(self, randomized: bool, headless: bool = False) -> Union[Environment, DomainRandomizationEnvironment]:
        """
        Returns an RLBench environment with consistent action mode, observation RLBench observation config,
        and domain randomization config.

        :param randomized: If true will return an environment with domain randomization
        :param headless:   If true will run headless (i.e. no display)

        :return: Either a normal or domain randomized RLBench environment
        """
        if not randomized:
            return Environment(action_mode=self.rlbench_actionmode,
                               obs_config=self.rlbench_obsconfig,
                               headless=headless)
        else:
            return DomainRandomizationEnvironment(action_mode=self.rlbench_actionmode,
                                                  obs_config=self.rlbench_obsconfig,
                                                  headless=headless,
                                                  visual_randomization_config=self.rlbench_random_config,
                                                  randomize_every=RandomizeEvery.EPISODE)

    def set_action_mode(self, mode: str):
        """
        Setter for the config's RLBench action mode. May be set to either absolute joint position
        or absolute joint velocity.

        :param mode: What mode to use. Should be either 'positions' or 'velocities'
        """
        if mode == 'positions':
            self.rlbench_actionmode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
        elif mode == 'velocities':
            self.rlbench_actionmode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        else:
            raise ValueError(f'Mode must be one of "positions" or "velocities"')

    def get_new_network(self, training_info: TrainingInfo) -> Tuple[Model, NetworkInfo]:
        """ Uses NetworkBuilder to generate a desired new network.

        :param training_info: String name of the training directory. This data is copied into the
                              returned NetworkInfo object so the input parameter may be deleted after.

        :returns: compiled network and network's metainformation
                  and training information as a NetworkInfo object
        """

        print('\nPlease enter the parameters for your network...')
        deep = bool(input('Use deep networks for gripper and joint inputs (yes/no. Default no): ') or False)
        deep = False if deep != 'yes' else True
        deep_option = '' if deep else 'not '
        print(f'[Info] Network will {deep_option}use deep networks for the gripper and joint inputs.')

        num_images = int(input('\nHow many image should the network use as an input (default 4): ') or 4)
        print(f'[Info] Network will accept {num_images} images as an input.')

        num_joints = int(input('\nHow man joints does your robot have (default 7 for Panda): ') or 7)
        print(f'[Info] Network will accept {num_joints} joint values as an input.')

        predict_mode = int(input('\nEnter 0 for position control or 1 for velocity control (default 0): ') or 0)
        predict_mode = 'velocities' if predict_mode == 1 else 'position'
        print(f'[Info] Network will predict {predict_mode} for the {num_joints} joints/')
        self.set_action_mode(predict_mode)

        print('\nPlease enter some training parameters for the network...')
        pov = self.get_pov_from_user()

        train_dir = training_info.train_dir.split('/')
        task, rand = self.get_task_name(train_dir[-3])

        print(f"\n[Info] Detected that the dataset's task is {task}.")

        rand_option = '' if rand else 'not '
        print(f"\n[Info] Detected that the dataset is {rand_option}randomized.")

        builder = NetworkBuilder(task=task,
                                 deep=deep,
                                 num_images=num_images,
                                 num_joints=num_joints,
                                 predict_mode=predict_mode,
                                 pov=pov,
                                 rand=rand)

        network = builder.get_network()

        # Part of the network_info is generated here
        network_info = builder.get_metainfo()
        network_info.predict_mode = predict_mode

        # For the rest of the info we pass the training_info data through to network_info.
        # After this all data is filled in network_info.
        network_info.transfer_training_info(training_info)

        return network, network_info

    def list_trained_networks(self) -> None:
        """ Prints a numbered list of all trained networks in the network
        directory with the number of episodes they contain. Saves
        this list in a self._possible_network. Network sets are selected
        from the list in self.load_network()

        :return: None
        """
        i = 0  # Only count if the item in network_root is a folder with children
        print(f'\nNetworks from the following directories may be used: ')
        for sub_dir in self.network_sub_dir:
            for net_dir in alpha_numeric_sort(listdir(join(self.network_root, sub_dir))):
                try:
                    _ = listdir(join(self.network_root, sub_dir, net_dir))
                    self._possible_network.append(join(sub_dir, net_dir))
                    print('{:.<20s}{:.<20s}'.format(f'Directory {i}',
                                                    f'{self._possible_network[i]}'))
                    i += 1
                except NotADirectoryError:
                    pass

    def get_trained_network_dir(self) -> str:
        """ Prints a numbered list of all trained networks in the network
        directory with the number of episodes they contain. User selects what
        directory/network to use and a path to the selection is returned

        :return: network_dir
        """
        self.list_trained_networks()

        try:
            network_num = int(input('\nEnter directory # of desired network: '))

            if network_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            network_dir = join(self.network_root, self._possible_network[network_num])
            return network_dir
        except (ValueError, IndexError):
            print(self._possible_network)
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    @staticmethod
    def load_trained_network(network_dir) -> Tuple[Model, NetworkInfo]:
        """
        Loads a tensorflow model from a specified directory.

        Assumes that the network file is .h5 format AND has the same name as the directory that
        is is placed in.

        :param network_dir: Path to the network directory

        :returns: A tuple with the TensorFlow model and its NetworkInfo object
        """
        network = load_model(join(network_dir, network_dir.split('/')[-1] + '.h5'))

        pickle_location = join(network_dir, 'network_info.pickle')
        with open(pickle_location, 'rb') as handle:
            network_info: NetworkInfo = pickle.load(handle)
        return network, network_info

    def list_data_set_directories(self) -> None:
        """ Prints a numbered list of all data sets in the data
        directory with the number of episodes they contain. Saves
        this list in a self._possible_data_set. Data sets are selected
        from the list in other functions, however.

        :return: None
        """
        i = 0  # Only count if the item in dataset_root is a folder with children
        print(f'\nThe data from the following directories may be used: ')

        for folder in alpha_numeric_sort(listdir(join(self.data_root))):
            try:
                for data in listdir(join(self.data_root, folder)):
                    self.possible_data_set.append(join(folder, data))
                    try:
                        num = len(listdir(join(self.data_root, self.possible_data_set[i], 'variation0', 'episodes')))
                    except FileNotFoundError:
                        num = 'NONE'
                    print('{:.<20s}{:.<20s}{:.<5s}'.format(f'Directory {i}',
                                                           f'{num} episodes',
                                                           f'{self.possible_data_set[i]}'))
                    i += 1
            except NotADirectoryError:
                pass

    def get_data_set_directory(self, prompt: str = 'Select a directory # to use: ') -> str:
        """
        Lists available datasets and returns the directory for just one.

        :param prompt: What to say in the user prompt for an input.

        :return: Path to dataset as {root}/{selected dataset}/variation0/episodes
        """
        self.list_data_set_directories()
        dir_num = int(input(f'\n{prompt}'))
        if (type(dir_num) is not int) or (dir_num < 0) or (dir_num > len(self.possible_data_set)):
            exit('\n[Error] Please enter a valid index above zero.')

        dataset_dir = join(self.data_root,
                           self.possible_data_set[dir_num],
                           'variation0',
                           'episodes')
        return dataset_dir

    def get_train_test_directories(self) -> Tuple[str, str]:
        """ Prints a numbered list of all data sets in the data
        directory with the number of episodes they contain. Users
        select what directories to use and a path to the selections
        is returned.

        :return: (train_dir, test_dir)
        """

        self.list_data_set_directories()

        try:
            train_num = int(input('\nEnter directory # for training: '))
            test_num = int(input('Enter directory # for testing: '))

            if train_num == test_num:
                input('\n[Warn] Testing and training on the same directory. This is not recommended. '
                      'Press enter to continue... ')
            elif train_num < 0 or test_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            train_dir = join(self.data_root, self.possible_data_set[train_num], 'variation0', 'episodes')
            test_dir = join(self.data_root, self.possible_data_set[test_num], 'variation0', 'episodes')

            return train_dir, test_dir
        except (ValueError, IndexError):
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def get_evaluate_directory(self) -> Tuple[str, int, int]:
        """ Prints a numbered list of all data sets in the data
        directory with the number of episodes they contain. Users
        select what directories to use and a path to the selections,
        the amount of evaluation episodes to use, and the amount of evaluation
        episodes available is returned.

        :return: (evaluation directory, evaluation amount, evaluation available)
        """
        self.list_data_set_directories()

        try:
            evaluation_num = int(input('\nEnter directory # for evaluation: '))

            if evaluation_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            eval_dir = join(self.data_root, self.possible_data_set[evaluation_num], 'variation0', 'episodes')

            available = len(listdir(eval_dir))
            print(f'\nThere are {available} episodes available for evaluation at '
                  f'{self.possible_data_set[evaluation_num]}')
            eval_amount = int(input('Enter how many to use (or -1 for all): '))
            if eval_amount > available or eval_amount <= -1:
                print(f'[Info] Using all {available} evaluation episodes.')
                eval_amount = available

            return eval_dir, eval_amount, available
        except (ValueError, IndexError):
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    @staticmethod
    def get_episode_amounts(train_dir: str, test_dir: str) -> TrainingInfo:
        """ Assists in getting and checking the number of training and testing demos to use
        and gets the number of training epochs. Also returns how many episodes are available in
        each directory.

        :param train_dir: full directory to training episodes
        :param test_dir:  full directory to testing episodes

        :return: A TrainingInfo object with the number of epochs and the directories, amount, and avalable episodes
                 for training and testing.
        """

        info = TrainingInfo()

        text = ['training', 'testing']
        amounts = [0, 0]
        default = [-1, 0]
        available = [0, 0]
        for i, d in enumerate([train_dir, test_dir]):
            available[i] = len(listdir(d))
            print(f'\nThere are {available[i]} episodes available for {text[i]} at {d}')
            to_use = int(input(f'Enter how many to use (or -1 for all. Default is {default[i]}): ') or default[i])
            if to_use > available[i] or to_use <= -1:
                to_use = available[i]
            amounts[i] = to_use
            print(f'[Info] Using {amounts[i]} {text[i]} episodes.')

        epochs = int(input('\nEnter how many epochs use (default is 1): ') or 1)
        if epochs < 1:
            epochs = 1
        print(f'[Info] Training for {epochs} epoch')

        info.train_dir = train_dir
        info.train_amount = amounts[0]
        info.train_available = available[0]

        info.test_dir = test_dir
        info.test_amount = amounts[1]
        info.test_available = available[1]

        info.epochs = epochs

        # train_dir is 'data' / 'tag' + {'_randomized'} / 'task' / 'variation0' / episodes
        #                                ^ iff randomized data
        pared_name = train_dir.split('/')
        info.task_name = pared_name[-3]
        info.randomized = True if 'randomized' in pared_name[-4].split('_') else False

        return info


if __name__ == '__main__':
    # Todo re-write this
    print('[Info] Starting config.py')

    e = EndToEndConfig()

    print(f'\nRunning this program displays the some common settings used across the '
          f'files in this directory. Running this file displays these settings '
          f'then tests some of the functions. \n')

    print(f'\n[Info] The root for all datasets is in: {e.data_root}\n')

    print(f'\n[Info] The root for all networks is in: {e.network_root}\n'
          f'[Info] There the following sub directories are used: {e.network_sub_dir}')

    print(f'\n[Info] The following custom network options are available: {e.custom_networks.keys()}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how testing and training datasets are selected: ')
    tr, te = e.get_train_test_directories()
    ntr, atr, nte, ate, epc = e.get_episode_amounts(tr, te)

    print(f'\n[Info] Training selection: {tr}')
    print(f'[Info] Training amount: {ntr} of {atr}')
    print(f'[Info] Training epochs: {epc}')
    print(f'[Info] Testing selection: {te} of {ate}')
    print(f'[Info] Testing amount: {nte}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how evaluation datasets are selected: ')
    ev, evn, eva = e.get_evaluate_directory()

    print(f'\n[Info] Evaluation selection: {ev}')
    print(f'[Info] Evaluation amount: {evn} of {eva}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how a new network is created')
    nn, _, _ = e.get_new_network()
    print(f'\n[Info] Network name: {nn}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how a network is created')
    nd, nm = e.get_trained_network()
    print(f'\n[Info] Network selection: {nd}')
    print(f'[Info] Network name: {nm}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how a task selected by the user')
    tn, tc = e.get_task_from_user()
    print(f'\nTask "{tn}" was selected with class {type(tc)}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is a successful get task from name selected by the user')
    tn, tc = e.get_task_from_name(['a', 'b', 'ReachTarget'])
    print(f'\nTask "{tn}" was selected with class {type(tc)}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is a failed get task from name selected by the user')
    tn, tc = e.get_task_from_name(['a', 'b', 'c'])
    print(f'\nTask "{tn}" was selected with class {type(tc)}')

    # pov from name
    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is a successful get poc from name selected by the user')
    p = e.get_pov_from_name(['a', 'b', 'wrist'])
    print(f'\nPOV "{p}" was selected')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is a failed get task from name selected by the user')
    p = e.get_pov_from_name(['a', 'b', 'c'])
    print(f'\nPOV "{p}" was selected')
