from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Softmax

from tensorflow.keras.models import Model

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy

from tensorflow.keras.optimizers import Adam

from distutils.dir_util import copy_tree

from os.path import join

# Todo find a better way to handle these conditional imports
try:
    from utils.network_info import NetworkInfo
except ModuleNotFoundError:
    from network_info import NetworkInfo


class NetworkBuilder(object):
    """ Creates network and saves metadata """
    def __init__(self, task: str, deep: bool = False, num_images: int = 4, num_joints: int = 7,
                 predict_mode: str = 'position', pov: str = 'front', rand: bool = False) -> None:
        """
        Takes users parameters for a network to create and compile the desired network.

        :param task:         String name of the task the network will be trained on
        :param deep:         True := use dense layers in the gripper and joint network.
                             False := pass gripper and joint values through to combined model
        :param num_images:   Number of images for the CNN to accept as an input
        :param num_joints:   Adjusts joint input to match the number of joints the robot has. The default (7)
                             matches the Franka Panda. But other robots (like the UR5) have 6 or fewer joints.
        :param predict_mode: What the joint output layer should predict. Either 'positions' or 'velocities'
        :param pov:          Point of view that the network will be trained for
        :param rand:         If the network is trained on randomized data
        """
        self._task = task
        self._deep = deep
        self._num_images = num_images
        self._num_joints = num_joints

        if predict_mode in ['positions', 'velocities']:
            self._predict_mode = predict_mode
        else:
            raise ValueError(f'Invalid control type "{predict_mode}" for NetworkBuilder.')

        self._pov = pov
        self._rand = rand

        self._name = None

        self.network = self._build_and_compile_network()
        self.stateful_network = self._build_and_compile_network(stateful=True)

    def _joints(self, stateful: bool = False) -> Model:
        """
        Creates a model for a robots joints. If a deep model is requested dense layer(s) are added.
        Otherwise, the model only flattens the input (e.g. does nothing) so a model may still be
        returned.

        Current intent is for joint angles but could be velocities or any other
        value. This is decided at the training stage.
        """
        batch_size = 1 if stateful else None
        pos_input = Input(shape=self._num_joints, batch_size=batch_size)
        if self._deep:
            pos = Dense(16, activation="tanh")(pos_input)
        else:
            pos = Flatten()(pos_input)
        return Model(inputs=pos_input, outputs=pos)

    def _gripper(self, stateful: bool = False) -> Model:
        """
        Creates a model for a robots gripper. If a deep model is requested dense layer(s) are added.
        Otherwise, the model only flattens the input (e.g. does nothing) so a model may still be
        returned.

        Current intent is for this to be categorical where 0 := close gripper and 1 := open gripper.
        However, this is decided at the compilation and training stage.
        """
        batch_size = 1 if stateful else None
        gripper_input = Input(shape=1, batch_size=batch_size)
        if self._deep:
            grip = Dense(4, activation="tanh")(gripper_input)
        else:
            grip = Flatten()(gripper_input)
        return Model(inputs=gripper_input, outputs=grip)

    def _cnn(self, stateful: bool = False) -> Model:
        """
        Creates a cnn to handle the depth camera input with the number of images requested.

        May be expanded with other CNN structures in the future.
        """
        batch_size = 1 if stateful else None
        vis_input = Input(shape=(128, 128, 4*self._num_images), batch_size=batch_size)

        # Todo: consider using other values/structures here...
        #       Currently using network inspired by:
        #       https://arxiv.org/abs/1707.02267
        filters = (32, 48, 64, 128, 192, 256, 256)
        size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (2, 2)]
        stride = [2, 2, 2, 2, 2, 2, 2]

        vis = vis_input
        for (i, f) in enumerate(filters):
            vis = Conv2D(f, size[i], strides=stride[i], padding="same")(vis)
            vis = Activation("relu")(vis)
            vis = BatchNormalization(axis=-1)(vis)

        vis = Flatten()(vis)
        vis = Dense(256)(vis)
        vis = Activation("relu")(vis)
        vis = BatchNormalization(axis=-1)(vis)
        vis = Activation("relu")(vis)

        return Model(inputs=vis_input, outputs=vis)

    def _build_and_compile_network(self, stateful: bool = False) -> Model:
        """
        Builds the multi-input-multi-output (mimo) model and stores it in the
        self.network member variable.

        :return: a compile network
        """
        joint_model = self._joints(stateful=stateful)
        grip_model = self._gripper(stateful=stateful)
        vis_model = self._cnn(stateful=stateful)

        combined = Concatenate()([joint_model.output,
                                  grip_model.output,
                                  vis_model.output])

        z = Reshape((1, combined.shape[1]))(combined)
        z = LSTM(128, stateful=stateful)(z)
        z = Dense(128, activation="relu")(z)

        #########
        # Names #
        #########
        output_joints_name = f'output_joints_{self._predict_mode}'
        output_action_name = f'output_action'
        output_target_name = f'output_target'
        output_gripper_name = f'output_gripper'

        #################
        # Output layers #
        #################

        output_joints = Dense(self._num_joints, activation="linear",
                              name=output_joints_name)(z)   # Joint values (e.g. angles or velocity), continuous
        output_action = Dense(2, activation="linear",)(z)
        output_action = Softmax(name=output_action_name)(output_action)    # Gripper action, categorical with
        output_target = Dense(3, activation="linear",
                              name=output_target_name)(z)   # Target (e.g. a cup) Cartesian position, continuous
        output_gripper = Dense(3, activation="linear",
                               name=output_gripper_name)(z)  # Gripper Cartesian position, continuous

        network = Model(inputs=[joint_model.input,
                                grip_model.input,
                                vis_model.input],
                        outputs=[output_joints,
                                 output_action,
                                 output_target,
                                 output_gripper])

        opt = Adam(learning_rate=0.0001)
        network.compile(optimizer=opt,
                        loss={output_joints_name: MeanSquaredError(),
                              output_action_name: MeanSquaredError(),
                              output_target_name: MeanSquaredError(),
                              output_gripper_name: MeanSquaredError(),
                              },
                        loss_weights=[1, 1, 1, 1],
                        metrics={output_joints_name: RootMeanSquaredError(),
                                 output_action_name: CategoricalAccuracy(),
                                 output_target_name: RootMeanSquaredError(),
                                 output_gripper_name: RootMeanSquaredError(),
                                 },
                        )

        name = [f'j{self._num_joints}g']
        if self._deep:
            name.append(f'deep')
        name.append(f'v{self._num_images}')
        name.append(f'{self._pov}')
        if self._rand:
            name.append(f'rand')
        name.append(f'{self._task}')
        name.append(f'{self._predict_mode}')
        self._name = '_'.join(name)

        return network

    def get_name(self) -> str:
        """
        Returns the network name. The name encodes information about the structure like if
        the network uses deep (dense) layer on the joint and gripper input and how many images are
        used.

        In training the name is appended further with information about the data used.

        Examples:
        jg_deep_v4  ->  Joint and gripper have deep layers and CNN accepts 4 images
        jg_v7       ->  Joint and gripper pass through and CNN accepts 7 images
        """
        return self._name

    def get_network(self) -> Model:
        """
        Getter function for the compiled network
        """
        return self.network

    def get_stateful_network(self) -> Model:
        """
        Getter function for the compiled  stateful network
        """
        return self.stateful_network

    def get_metainfo(self) -> NetworkInfo:
        """
        Returns dictionary with meta information about the network. Information is encoded in the
        name too but this makes it simpler to access.
        """
        info = NetworkInfo()

        info.network_name = self.get_name()
        info.deep = self._deep
        info.num_joints = self._num_joints
        info.num_images = self._num_images

        info.randomized = self._rand
        info.pov = self._pov

        info.task_name = self._task

        return info


class StatelessToStateful(object):
    def __init__(self, stateless_model: Model, network_info: NetworkInfo, network_dir: str):
        self._stateless_model = stateless_model
        self._network_info = network_info

        self._builder = NetworkBuilder(task=network_info.task_name,
                                       deep=network_info.deep,
                                       num_images=network_info.num_images,
                                       num_joints=network_info.num_joints,
                                       predict_mode=network_info.predict_mode,
                                       pov=network_info.pov,
                                       rand=network_info.randomized)

        self._stateful_model = self._builder.get_stateful_network()
        self._stateful_model.set_weights(self._stateless_model.get_weights())

        self._network_dir = network_dir

    def save(self):
        new_network_dir = self._network_dir + '_stateful'
        copy_tree(src=self._network_dir, dst=new_network_dir)

        new_network_name = self._network_info.network_name + '_stateful.h5'
        self._stateful_model.save(join(new_network_dir, new_network_name))


def main():
    print('This script is for testng the NetworkBuilder class. Please define your network...')
    num_joints = int(input('How many joint inputs: '))
    num_images = int(input('How many image inputs: '))
    deep = input('Use deep networks for the gripper and joint branches (y/n): ')
    deep = True if deep == 'y' else False
    task = input('Type the name of your task: ')

    builder = NetworkBuilder(deep=deep,
                             num_images=num_images,
                             num_joints=num_joints,
                             task=task)

    network = builder.get_network()
    name = builder.get_name()
    info = builder.get_metainfo()

    print(f'Network is: {network}')
    print(f'Network named: {name}')
    print(f'Network meta information... \n'
          f'{info}')


if __name__ == "__main__":
    main()
