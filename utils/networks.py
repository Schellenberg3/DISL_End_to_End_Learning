from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape


class NetworkBuilder(object):
    """ Creates network and saves metadata """
    def __init__(self, deep: bool = False, num_images: int = 4, num_joints: int = 7) -> None:
        """
        Takes users parameters for a network to create and compile the desired network.

        :param deep:       True := use dense layers in the gripper and joint network.
                           False := pass gripper and joint values through to combined model
        :param num_images: Number of images for the CNN to accept as an input
        :param num_joints: Adjusts joint input to match the number of joints the robot has. The default (7)
                           matches the Franka Panda. But other robots (like the UR5) have 6 or fewer joints.
        """
        self._deep = deep
        self._num_images = num_images
        self._num_joints = num_joints

        self._name = None
        self._split = None

        self.network = self._build_and_compile_network()

    def _joints(self) -> Model:
        """
        Creates a model for a robots joints. If a deep model is requested dense layer(s) are added.
        Otherwise, the model only flattens the input (e.g. does nothing) so a model may still be
        returned.

        Current intent is for joint angles but could be velocities or any other
        value. This is decided at the training stage.
        """
        pos_input = Input(shape=self._num_joints)
        if self._deep:
            pos = Dense(16, activation="tanh")(pos_input)
        else:
            pos = Flatten()(pos_input)
        return Model(inputs=pos_input, outputs=pos)

    def _gripper(self) -> Model:
        """
        Creates a model for a robots gripper. If a deep model is requested dense layer(s) are added.
        Otherwise, the model only flattens the input (e.g. does nothing) so a model may still be
        returned.

        Current intent is for this to be categorical where 0 := close gripper and 1 := open gripper.
        However, this is decided at the compilation and training stage.
        """
        gripper_input = Input(shape=1)
        if self._deep:
            grip = Dense(4, activation="tanh")(gripper_input)
        else:
            grip = Flatten()(gripper_input)
        return Model(inputs=gripper_input, outputs=grip)

    def _cnn(self) -> Model:
        """
        Creates a cnn to handle the depth camera input with the number of images requested.

        May be expanded with other CNN structures in the future.
        """
        vis_input = Input(shape=(128, 128, 4*self._num_images))

        # Todo: consider using other values/structures here...
        #       Currently using network inspired by:
        #       https://arxiv.org/abs/1707.02267
        filters = (32, 48, 64, 128, 192, 256, 256)
        size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (2, 2)]
        stride = [2, 2, 2, 2, 2, 2, 2]

        for (i, f) in enumerate(filters):
            vis = Conv2D(f, size[i], strides=stride[i], padding="same")(vis_input)
            vis = Activation("relu")(vis)
            vis = BatchNormalization(axis=-1)(vis)

        vis = Flatten()(vis)
        vis = Dense(256)(vis)
        vis = Activation("relu")(vis)
        vis = BatchNormalization(axis=-1)(vis)
        vis = Dropout(0.5)(vis)  # Todo remove this
        vis = Activation("relu")(vis)

        return Model(inputs=vis_input, outputs=vis)

    def _build_and_compile_network(self) -> Model:
        """
        Builds the multi-input-multi-output (mimo) model and stores it in the
        self.network member variable.

        :return: a compile network
        """
        joint_model = self._joints()
        grip_model = self._gripper()
        vis_model = self._cnn()

        combined = Concatenate()([joint_model.output,
                                  grip_model.output,
                                  vis_model.output])

        z = Reshape((1, combined.shape[1]))(combined)
        z = LSTM(128)(z)
        z = Dense(128, activation="relu")(z)

        output_joints = Dense(7, activation="linear")(z)   # Joint values (e.g. angles or velocity), continuous
        output_action = Dense(1, activation="linear")(z)   # Gripper action, categorical
        output_target = Dense(3, activation="linear")(z)   # Target (e.g. a cup) Cartesian position, continuous
        output_gripper = Dense(3, activation="linear")(z)  # Gripper Cartesian position, continuous

        network = Model(inputs=[joint_model.input,
                                grip_model.input,
                                vis_model.input],
                        outputs=[output_joints,
                                 output_action,
                                 output_target,
                                 output_gripper])

        network.compile(optimizer='adam',
                        loss={'output_joint_angles': 'cosine_similarity',
                              'output_gripper_state': 'sparse_categorical_crossentropy',
                              'output_gripper_pose': 'mse',
                              'output_target_pose': 'mse',
                              }
                        )

        name = ['jg_']
        if self._deep:
            name.append(f'deep')
        name.append(f'v{self._num_images}')
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
        if self._name:
            return self._name
        else:
            raise RuntimeError('Network must be built before it has a name')

    def get_network(self) -> Model:
        """
        Getter function for the compiled network
        """
        return self.network

    def get_metainfo(self) -> dict:
        """
        Returns dictionary with meta information about the network. Information is encoded in the
        name too but this makes it simpler to access.
        """
        info = {'name': self._name,
                'num_images': self._num_images,
                'deep': self._deep}
        return info


if __name__ == "__main__":
    print("This script contains the basic network configurations considered in the research project.\n"
          "They are mostly permutations of the same basic model\n")

    use_optimizer = "adam"
    use_loss = "mean_squared_error"
    use_metrics = ["accuracy", "mse"]

    cnn_settings = "James"

    save_dir = 'network_info'

    networks = {
        "rnn_pv": rnn_position_vision,
        "pv": position_vision,
        "rnn_pv4": rnn_position_vision_4,
        "pv4": position_vision_4,
    }

    for model_name in networks:
        save_network_info(model_name, cnn_settings, networks[model_name], save_dir)
