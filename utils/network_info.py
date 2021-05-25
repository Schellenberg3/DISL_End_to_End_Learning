
class NetworkInfo(object):
    """ Storage object for information about a network's settings and previous training """

    def __init__(self):
        # Network's settings
        self.network_name = None
        self.deep = None
        self.num_joints = None
        self.num_images = None
        self.pov = None

        # Information passed from a TrainingInfo object
        self.task_name = None
        self.randomized = None

        self.train_dir = None
        self.test_dir = None

        self.train_amount = None
        self.test_amount = None

        self.train_available = None
        self.test_available = None

        self.epochs_to_train = 0
        self.prev_epochs = 0
        self.total_epochs = 0

    def __str__(self):
        return f'Network Name: {self.network_name}\n' \
               f'Network is deep: {self.deep}\n' \
               f'Network accepts: {self.num_images} image and {self.num_joints} joints\n' \
               f'Trained for the {self.task_name} task from {self.pov} point of view\n' \
               f'Randomized training data: {self.randomized}\n' \
               f'Trained for: {self.train_amount} episodes over {self.prev_epochs} epochs\n' \
               f'Will train for {self.epochs_to_train} more epochs on {self.train_available} ' \
               f'episodes from {self.train_dir}\n' \
               f'Testing amount is {self.test_amount} of {self.test_available} episodes in ' \
               f'{self.test_dir}'
