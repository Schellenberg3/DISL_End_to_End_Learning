


# todo consider moving to separate files
#    - utils and maybe custom_networks go to assets?
#    - use if __name__ == 'imitation_learner' to import TensorFlow only when needed? (NameError if module isn't defined)
class EndToEndConfig:

    def __init__(self):
        # DO NOT MOVE THIS FILE FROM THE MAIN FOLDER. WILL BREAK DIRECTORY LOCATION ASSUMPTION
        self.network_root = join(os.path.dirname(os.path.realpath(__file__)), 'trained_networks')
        self.data_root = join(os.path.dirname(os.path.realpath(__file__)), 'data')
        # todo
        self.networks_root = join(os.path.dirname(os.path.realpath(__file__)), 'networks')
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

    # todo update reference in imitation_learner
    def get_new_network(self):
        """
        Lists network options from custom_networks and lets user choose pick a
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
                exit('[ERROR] Selections must be integers. Exiting program')
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

    # todo write code to get a saved model

    def set_directories(self):
        """
        Lists the directories the data/ and asks the user to pick which
        on to use for testing and training

        :return: Tuple[train_dir, test_dir]
        """
        possible_data_set = []
        i = 0  # not enumerate -only count if the item in dataset_root is a folder with children
        print(f'\nThe data from the following directories may be used: ')
        for folder in listdir(join(self.data_root)):
            try:
                for data in listdir(join(self.data_root, folder)):
                    possible_data_set.append(join(folder, data))
                    try:
                        num = len(listdir(join(self.data_root, possible_data_set[i], 'variation0', 'episodes')))
                    except FileNotFoundError:
                        num = 'NONE'
                    print('{:.<20s}{:.<20s}{:.<5s}'.format(f'Directory {i}',
                                                           f'{num} episodes',
                                                           f'{possible_data_set[i]}'))
                    i += 1
            except NotADirectoryError:
                pass

        try:
            train_num = int(input('\nEnter directory # for training: '))
            test_num = int(input('Enter directory # for testing: '))

            if train_num == test_num:
                exit('\n[ERROR] Cannot test and train on the same directory. Exiting program.')
            elif train_num < 0 or test_num < 0:
                exit('\n[ERROR] Selections must be greater than zero. Exiting program.')

            train_dir = join(self.data_root, possible_data_set[train_num], 'variation0', 'episodes')
            test_dir = join(self.data_root, possible_data_set[test_num], 'variation0', 'episodes')

            return train_dir, test_dir
        except (ValueError, IndexError) as e:
            exit('\n[ERROR] Selections must be integers and valid list indices. Exiting program')

    def get_episode_amounts(self, train_dir, test_dir):
        """
        Assists in getting and checking the number of training and testing demos to use
        and gets the number of training epochs

        :param train_dir: full directory to training episodes
        :param test_dir: full directory to testing episodes
        :return: Tuple[num_test, num_train, epochs
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

        return amounts, epochs