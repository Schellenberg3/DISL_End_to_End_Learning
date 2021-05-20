from rlbench.tasks import DislPickUpBlueCup
from rlbench.tasks import ReachTarget

from utils.networks import NetworkBuilder
from utils.utils import alpha_numeric_sort
from utils.utils import split_data
from utils.utils import check_yes

from tensorflow.keras import Model

from os.path import dirname
from os.path import realpath
from os.path import join
from os.path import isdir
from os import listdir

from typing import Tuple
from typing import Dict
from typing import List
from typing import Any


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

    def get_pov_from_user(self) -> str:
        """ If point of view cannot be inferred from the network's name
        this prompts the user to select which camera view to use.

        :return: Either "wrist" or "front"
        """
        pov = input('Please enter what camera point of view to use, front (default) or wrist: ')
        if pov in ['front', 'Front', 'wrist', 'Wrist']:
            return pov.lower()
        else:
            print(f'[Warn] Input "{pov}" is not an option, will use default point of view: front')
            return 'front'

    def get_new_network(self) -> Tuple[Model, str, Dict]:
        """ Uses NetworkBuilder to generate a desired new network.

        :returns: compiled network, network's name, and network's metainformation
        """

        print('\nPlease enter the parameters for your network...')
        deep = bool(input('Use deep networks for gripper and joint inputs (y/n): ') or False)
        num_images = int(input('How many image should the network use as an input (default 4): ') or 4)
        num_joints = int(input('How man joints does your robot have (default 7 for Panda): ') or 7)

        builder = NetworkBuilder(deep=deep,
                                 num_images=num_images,
                                 num_joints=num_joints)

        name = builder.get_name()
        info = builder.get_metainfo()
        network = builder.get_network()

        return network, name, info


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

    def get_trained_network(self) -> Tuple[str, str]:
        """ Prints a numbered list of all trained networks in the network
        directory with the number of episodes they contain. User selects what
        directory/network to use and a path to the selection is returned

        :return: (network_dir, network_name)
        """
        self.list_trained_networks()

        try:
            network_num = int(input('\nEnter directory # of desired network: '))

            if network_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            network_dir = join(self.network_root, self._possible_network[network_num])
            _, network_name = self._possible_network[network_num].split('/')
            return network_dir, network_name
        except (ValueError, IndexError) as e:
            print(self._possible_network)
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def get_info_from_network_name(self, parsed_name: List[str]):
        """ For retraining a network, this takes the name of a trained network and
        returns the point of view ('front' or 'wrist'), what function to use to split
        data, the task's training directory (testing directory is returned for consistency
        but this should not be used in retraining), the RLBench class for the task, the
        the amount of training episodes to uses (which is equal to the number available in the
        training directory), the amount of training episodes available, the amount of testing episodes
        to use (included for consistency but set to 0), the amount of testing episodes available
        (included for consistency but set to 0), and the number of epochs previously trained over.

        :param parsed_name: Network name split
        :return: (Point of View, split function, task's name, RLBench task class,
        training directory, testing directory, train amount, train available,
        test amount, test available, previous epochs)
        """

        pov = self.get_pov_from_name(parsed_name)

        task_name, task = self.get_task_from_name(parsed_name)

        if 'rnn-pv4' in parsed_name or 'pv4' in parsed_name:
            split = split_data_4
            print(f'\n[Info] Detected that the network uses 4 images, will format images accordingly')
        else:
            split = split_data

        epoch_info = [i for i in parsed_name if 'by' in i]
        if len(epoch_info) != 1:
            prev_epoch = int(input('\n[Warn] Could not infer the prior number of epochs.\n'
                                   'Please enter the number of epochs previously trained over: '))
        else:
            prev_epoch = int(epoch_info[0][2:])
            print(f'\n[Info] Detected that the network was trained over {prev_epoch} epochs')

        tag = 'training'
        tag += '_randomized' if 'rand' in parsed_name else ''

        train_dir = join(self.data_root,
                         tag,
                         task_name,
                         'variation0',
                         'episodes')

        if isdir(train_dir):
            train_available = len(listdir(train_dir))
            print(f'\n[Info] Automatically selected dataset: {join(tag, task_name)} \n'
                  f'[Info] Retraining will use all {train_available} episodes.')

            train_amount = train_available

            print('\n[Info] No testing will be performed. Use evaluate.py instead.')

            test_dir = train_dir
            test_amount = 0
            test_available = 0
        else:
            print(f'\n[Info] Failed to detect the correct training directory. Note that retraining '
                  f'uses all available episodes and does not do a final evaluation.\n '
                  f' Please select a directory: ')
            train_dir, _ = self.get_train_test_directories()

            train_available = len(listdir(train_dir))
            print(f'\n[Info] Retraining will use all {train_available} episodes.')

            train_amount = train_available

            print('\n[Info] No testing will be performed. Use evaluate.py instead.')

            test_dir = train_dir
            test_amount = 0
            test_available = 0

        return pov, split, task_name, task, train_dir, test_dir,\
            train_amount, train_available, \
            test_amount, test_available, prev_epoch

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
                print('\n[Warn] Testing and training on the same directory. This is not recommended.')
            elif train_num < 0 or test_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            train_dir = join(self.data_root, self.possible_data_set[train_num], 'variation0', 'episodes')
            test_dir = join(self.data_root, self.possible_data_set[test_num], 'variation0', 'episodes')

            return train_dir, test_dir
        except (ValueError, IndexError) as e:
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
        except (ValueError, IndexError) as error:
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def get_episode_amounts(self, train_dir: str, test_dir: str) -> Tuple[int, int, int, int, int]:
        """ Assists in getting and checking the number of training and testing demos to use
        and gets the number of training epochs. Also returns how many episodes are available in
        each directory.

        :param train_dir: full directory to training episodes
        :param test_dir: full directory to testing episodes
        :return: (train amount, train available, test amount, test available, epochs)
        """
        text = ['training', 'testing']
        amounts = [0, 0]
        available = [0, 0]
        for i, d in enumerate([train_dir, test_dir]):
            available[i] = len(listdir(d))
            print(f'\nThere are {available[i]} episodes available for {text[i]} at {d}')
            to_use = int(input('Enter how many to use (or -1 for all): '))
            if to_use > available[i] or to_use <= -1:
                print(f'[Info] Using all {available[i]} {text[i]} episodes.')
                to_use = available[i]
            amounts[i] = to_use

        epochs = int(input('\nEnter how many epochs use: '))
        if epochs < 1:
            print(f'[Info] Setting epochs to 1')
            epochs = 1

        return amounts[0], available[0], amounts[1], available[1], epochs


if __name__ == '__main__':
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