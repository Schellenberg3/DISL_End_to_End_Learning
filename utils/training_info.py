
class TrainingInfo(object):
    """ Storage object for information about a how a network will be trained """

    def __init__(self):
        self.train_dir = None
        self.test_dir = None

        self.train_amount = None
        self.test_amount = None

        self.train_available = None
        self.test_available = None

        self.epochs = None
