from utils.networks import position_vision
from utils.networks import position_vision_4
from utils.networks import rnn_position_vision
from utils.networks import rnn_position_vision_4
from utils.utils import split_data
from utils.utils import split_data_4
from utils.utils import check_yes
from os.path import dirname
from os.path import realpath
from os.path import join
from os import listdir
from typing import Tuple
from typing import List



class EndToEndConfig:

    def __init__(self):
        """ Container for common variables and getter/setters used
        throughout the code.
        """
        # DO NOT MOVE THIS FILE FROM THE MAIN FOLDER. WILL BREAK DIRECTORY LOCATION ASSUMPTION
        self.data_root = join(dirname(realpath(__file__)), 'data')
        self._possible_data_set = []

        self.network_root = join(dirname(realpath(__file__)), 'networks')
        self.network_sub_dir = ['imitation', 'reinforcement']
        self._possible_network = []

        # long name : short name, model function, related split function
        self.custom_networks = {"position_vision": ("pv",
                                                    position_vision,
                                                    split_data),
                                "position_vision_4": ("pv4",
                                                      position_vision_4,
                                                      split_data_4),
                                "rnn_position_vision": ("rnn-pv",
                                                        rnn_position_vision,
                                                        split_data),
                                "rnn_position_vision_4": ("rnn-pv4",
                                                          rnn_position_vision_4,
                                                          split_data_4),
                                }

        # todo: add selection of RLBench tasks
        #    - May need to edit how the networks are named for this
        #    - Should use a dict
        #    - Generator should call this to ensure naming convention (and data_root)
        self.tasks = {}

    def get_new_network(self):
        """ Lists network options from custom_networks and lets user choose pick a
        configuration. Returns the network's name, the network function, and
        supporting data split function.

        :return: Tuple[str, Model, function]
        """

        print('\nThe following networks are available:')
        list_keys = []
        for i, (k, v) in enumerate(self.custom_networks.items()):
            print(f'Option {i}.....{k}')
            list_keys.append(k)

        try:
            model_selection = int(input('\nPlease enter the option # for the network you would like to create: '))
            if model_selection > len(list_keys):
                exit('[ERROR] Selections must one of the listed options. Exiting program')
        except ValueError:
            exit('[ERROR] Selections must be integers. Exiting program')

        if check_yes('\nWill a data set with domain randomization be used in training? (y/n): '):
            domain = "rand"
        else:
            domain = "norm"

        cnn_setting = input('\nEnter 0 to use a James inspired CNN, 1 to use a Hermann inspired CNN, or '
                            'anything else to use the custom one: ')
        if cnn_setting == '0':
            cnn_setting = 'James'
        elif cnn_setting == '1':
            cnn_setting = "Hermann"
        else:
            cnn_setting = "Custom"

        key = list_keys[model_selection]
        name = f'{self.custom_networks[key][0]}_{domain}_{cnn_setting}'
        model = self.custom_networks[key][1](cnn_setting)
        split = self.custom_networks[key][2]

        print(f'\n[Info] Network will be {key} configured with {cnn_setting} CNN')

        return name, model, split

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
            for net_dir in listdir(join(self.network_root, sub_dir)):
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

        :return: network_dir, network_name
        """
        self.list_trained_networks()

        try:
            network_num = int(input('\nEnter directory # of desired network: '))

            if network_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            network_dir = join(self.network_root, self._possible_data_set[network_num])
            _, network_name = self._possible_data_set[network_num].split('/')[1]
            return network_dir, network_name
        except (ValueError, IndexError) as e:
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def list_data_set_directories(self) -> None:
        """ Prints a numbered list of all data sets in the data
        directory with the number of episodes they contain. Saves
        this list in a self._possible_data_set. Data sets are selected
        from the list in other functions, however.

        :return: None
        """
        i = 0  # Only count if the item in dataset_root is a folder with children
        print(f'\nThe data from the following directories may be used: ')
        for folder in listdir(join(self.data_root)):
            try:
                for data in listdir(join(self.data_root, folder)):
                    self._possible_data_set.append(join(folder, data))
                    try:
                        num = len(listdir(join(self.data_root, self._possible_data_set[i], 'variation0', 'episodes')))
                    except FileNotFoundError:
                        num = 'NONE'
                    print('{:.<20s}{:.<20s}{:.<5s}'.format(f'Directory {i}',
                                                           f'{num} episodes',
                                                           f'{self._possible_data_set[i]}'))
                    i += 1
            except NotADirectoryError:
                pass

    def get_test_train_directories(self) -> Tuple[str, str]:
        """ Prints a numbered list of all data sets in the data
        directory with the number of episodes they contain. Users
        select what directories to use and a path to the selections
        is returned.

        :return: Tuple[train_dir, test_dir]
        """

        self.list_data_set_directories()

        try:
            train_num = int(input('\nEnter directory # for training: '))
            test_num = int(input('Enter directory # for testing: '))

            if train_num == test_num:
                exit('\n[ERROR] Cannot test and train on the same directory. Exiting program.')
            elif train_num < 0 or test_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            train_dir = join(self.data_root, self._possible_data_set[train_num], 'variation0', 'episodes')
            test_dir = join(self.data_root, self._possible_data_set[test_num], 'variation0', 'episodes')

            return train_dir, test_dir
        except (ValueError, IndexError) as e:
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def get_evaluate_directory(self) -> Tuple[str, int]:
        """ Prints a numbered list of all data sets in the data
        directory with the number of episodes they contain. Users
        select what directories to use and a path to the selections
        is returned.

        :return: str
        """
        self.list_data_set_directories()

        try:
            evaluation_num = int(input('\nEnter directory # for evaluation: '))

            if evaluation_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            eval_dir = join(self.data_root, self._possible_data_set[evaluation_num], 'variation0', 'episodes')

            available = len(listdir(eval_dir))
            print(f'\nThere are {available} episodes available for evaluation at '
                  f'{self._possible_data_set[evaluation_num]}')
            eval_amount = int(input('Enter how many to use (or -1 for all): '))
            if eval_amount > available or eval_amount <= -1:
                print(f'[Info] Using all {available} evaluation episodes.')
                eval_amount = available

            return eval_dir, eval_amount
        except (ValueError, IndexError) as error:
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def get_episode_amounts(self, train_dir: str, test_dir: str) -> Tuple[int, int, int]:
        """ Assists in getting and checking the number of training and testing demos to use
        and gets the number of training epochs

        :param train_dir: full directory to training episodes
        :param test_dir: full directory to testing episodes
        :return: Tuple[num_test, num_train, epochs]
        """
        text = ['training', 'testing']
        amounts = [0, 0]
        for i, d in enumerate([train_dir, test_dir]):
            available = len(listdir(d))
            print(f'\nThere are {available} episodes available for {text[i]} at {d}')
            to_use = int(input('Enter how many to use (or -1 for all): '))
            if to_use > available or to_use <= -1:
                print(f'[Info] Using all {available} {text[i]} episodes.')
                to_use = available
            amounts[i] = to_use

        epochs = int(input('\nEnter how many epochs use: '))
        if epochs < 1:
            print(f'[Info] Setting epochs to 1')
            epochs = 1

        return amounts[0], amounts[1], epochs


# todo write a print statement for the config
if __name__ == '__main__':
    e = EndToEndConfig()

    print(f'Running this program displays the some common settings used across the '
          f'files in this directory. Running this file displays these settings '
          f'then tests some of the functions. \n')

    print(f'[Info] The root for all datasets is in: {e.data_root}\n')

    print(f'[Info] The root for all networks is in: {e.network_root}\n'
          f'[Info] There the following sub directories are used: {e.network_sub_dir}\n')

    print(f'[Info] The following custom network options are available: {e.custom_networks.keys()}\n')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how testing and training datasets are selected: ')
    tr, te = e.get_test_train_directories()
    ntr, nte, epc = e.get_episode_amounts(tr, te)

    print(f'\n[Info] Training selection: {tr}')
    print(f'[Info] Training amount: {ntr}')
    print(f'[Info] Testing selection: {te}')
    print(f'[Info] Testing amount: {nte}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how evaluation datasets are selected: ')
    ev, num = e.get_evaluate_directory()

    print(f'\n[Info] Evaluation selection: {ev}')
    print(f'[Info] Evaluation amount: {num}')

    print('\n-------------------------------------------------------\n')

    print(f'[Info] Here is how a new model is created')
    nn, _, _ = e.get_new_network()
    print(f'\n[Info] network name: {nn}')


