from os.path import isfile
from os.path import join
from os.path import isdir
from os import mkdir
from PIL import Image
import numpy as np


class Recorder(object):
    """ Class to store images from a demo and save them to a gif within the network's folder. """
    def __init__(self, network_dir: str):
        """
        This class handles the creation of recordings of the network. These are saved in the
        network's directory in a folder called 'recordings'

        :param network_dir: Directory where the network is stored.
        """
        self._network_dir = network_dir
        self._recording_dir = join(network_dir, 'recordings')
        if not isdir(self._recording_dir):
            mkdir(self._recording_dir)
        self._images = []

    def add_image(self, image: np.ndarray):
        """
        Adds an image to the recorder.

        :param image: the RGB image to add to the object.
        """
        self._images.append(Image.fromarray((image * 255).astype(np.uint8)))

    def clear_images(self):
        """
        Resets the images to a blank list so the same recorder may be used for
        multiple episodes in a row.
        """
        self._images = []

    def save_gif(self, fname: str = 'demo_gif', clear_after: bool = True):
        """
        Save the set of collected images in the networks directory.

        :param fname:       Name for the saved gif file. Does not need the extension '.gif'
        :param clear_after: If true will call self.clear_images to reset the recording object
        """
        save_as = join(self._recording_dir, f'{fname}.gif')

        i = 0
        while isfile(save_as):
            save_as = join(self._recording_dir, f'{fname}-{i}.gif')
            i += 1

        self._images[0].save(save_as,
                             save_all=True,
                             append_images=self._images[1:],
                             optimize=False,
                             duration=40,
                             loop=0)
        if clear_after:
            self.clear_images()

    def save_video(self):
        """
        Possible expansion of the class.
        """
        raise NotImplementedError
