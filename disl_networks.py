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


def pos_net():
    # Note that all position inputs have a shape of 8 -> [7 panda joints, gripper open/close]
    pos_input = Input(shape=8)
    pos = Dense(64, activation="tanh")(pos_input)
    pos = Dense(64, activation="tanh")(pos)
    pos = Dense(64, activation="tanh")(pos)
    return Model(inputs=pos_input, outputs=pos)


def vis_net(settings="Hermann", four_deep=False):
    if not four_deep:
        vis_input = Input(shape=(128, 128, 4))
    else:
        vis_input = Input(shape=(128, 128, 16))

    if settings == "Hermann":
        # Using a Hermann inspired network:
        #    https://arxiv.org/abs/1910.07972
        filters = (32, 64, 32)
        size = [(8, 8), (4, 4), (3, 3)]
        stride = [4, 2, 1]
    elif settings == "James":
        # Assume using a James inspired network:
        #    https://arxiv.org/abs/1707.02267
        filters = (32, 48, 64, 128, 192, 256, 256)
        size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (2, 2)]
        stride = [2, 2, 2, 2, 2, 2, 2]
    else:
        # Todo: decide if these should/could be custom values
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
    vis = Dropout(0.5)(vis)
    vis = Activation("relu")(vis)

    return Model(inputs=vis_input, outputs=vis)


def rnn_position_vision(vis_settings="Hermann"):
    pos_model = pos_net()
    vis_model = vis_net(settings=vis_settings)

    combined = Concatenate()([pos_model.output, vis_model.output])

    out = Reshape((1, combined.shape[1]))(combined)
    out = LSTM(128)(out)
    out = Dense(128, activation="relu")(out)
    out = Dense(8, activation="linear")(out)

    return Model(inputs=[pos_model.input, vis_model.input], outputs=out)


def rnn_vision(vis_settings="Hermann"):
    vis_model = vis_net(settings=vis_settings)

    out = Flatten()(vis_model.output)
    out = Reshape((1, out.shape[1]))(out)
    out = LSTM(128)(out)
    out = Dense(128, activation="relu")(out)
    out = Dense(8, activation="linear")(out)

    return Model(inputs=vis_model.input, outputs=out)


def rnn_position_vision_4(vis_settings="Hermann"):
    pos_model = pos_net()
    vis_model = vis_net(settings=vis_settings, four_deep=True)

    combined = Concatenate()([pos_model.output, vis_model.output])

    out = Reshape((1, combined.shape[1]))(combined)
    out = LSTM(128)(out)
    out = Dense(128, activation="relu")(out)
    out = Dense(8, activation="linear")(out)

    return Model(inputs=[pos_model.input, vis_model.input], outputs=out)


if __name__ == "__main__":
    print("This script contains the basic network configurations considered in the research project.\n"
          "They are mostly permutations of the same basic model\n")


    use_optimizer = "adam"
    use_loss = "mean_squared_error"
    use_metrics = ["accuracy", "mse"]

    cnn_settings = "James"

    vp_model = rnn_position_vision(cnn_settings)
    vp_model.compile(optimizer=use_optimizer,
                     loss=use_loss,
                     metrics=use_metrics)
    keras.utils.plot_model(vp_model, "images/vp_model.png", show_shapes=True)
    vp_model.summary()

    vp4_model = rnn_position_vision_4(cnn_settings)
    vp4_model.compile(optimizer=use_optimizer,
                      loss=use_loss,
                      metrics=use_metrics)
    keras.utils.plot_model(vp4_model, "images/vp4_model.png", show_shapes=True)
    vp4_model.summary()

    v_model = rnn_vision(cnn_settings)
    v_model.compile(optimizer=use_optimizer,
                    loss=use_loss,
                    metrics=use_metrics)
    keras.utils.plot_model(v_model, "images/v_model.png", show_shapes=True)
    v_model.summary()

