import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
import numpy as np
import matplotlib.pyplot as plt

input_gripper_state = Input(shape=1)
input_joint_angles = Input(shape=7)
input_depth_history = Input(shape=(128, 128, 16))


filters = (32, 48, 64, 128, 192, 256, 256)
# The size of the kernel window applied for each filter

# Herman
# size = [(8, 8), (4, 4), (3, 3)]

# James
size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (2, 2)]

# Stride defines how many pixels we move before the next convolution is applied
#    Practically, this means the dimensions of the output will be the input / stride
#    e.g. an 8x8 image after a stride of 2 will be 4x4
#    Stride may also be a tuple for the height and length dimensions
#
#    It seems that the stride is usually half the kernel window's dimensions

# Herman
# stride = [4, 2, 1]

# James
stride = [2, 2, 2, 2, 2, 2, 2]

chanDim = -1

image_net = input_depth_history
for (i, f) in enumerate(filters):
    # CONV => RELU => BN => POOL
    #
    # Convolution is the standard process of creating a sparsely connected network with shared weights/biases
    #    This is the standard way to process images with DNNs
    #    TF returns a 4+D tensor with shape: batch_shape + (new_rows, new_cols, filters)
    #
    # BatchNorm before non-linear activation
    #    reduces the "covariate shift" [3] or input distribution by keeping the mean and std constant.
    #    For CNN we normalize for each filter, stored in the last axis of the tensor (hence axis = -1)
    #    See: [1] https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
    #         [2] https://www.baeldung.com/cs/batch-normalization-cnn
    #         [3] https://arxiv.org/pdf/1502.03167.pdf
    #
    # Activation with ReLu will add non-linearity to the network allowing for a more complex fit
    #
    # MaxPooling "Downsamples the input representation by taking the maximum value over the window" FT API
    #    Reduces the size of the image each time.  Here is using (2,2) the image dimensions are halved
    image_net = Conv2D(f, size[i], strides=stride[i], padding="same", activation="relu")(image_net)
    image_net = BatchNormalization(axis=chanDim)(image_net)
    image_net = Activation("relu")(image_net)
    # y = MaxPooling2D((2, 2), padding="same")(y)  # neither group used max pooling

image_net = Flatten()(image_net)
image_net = Dense(16)(image_net)
image_net = Activation("relu")(image_net)
image_net = BatchNormalization(axis=chanDim)(image_net)
image_net = Dense(512)(image_net)
image_net = Activation("relu")(image_net)

image_net = Model(input_depth_history, image_net)


combined = Concatenate()([input_gripper_state, input_joint_angles, image_net.output])


z = Reshape((1, combined.shape[1]))(combined)
z = LSTM(128)(z)
z = Dense(128, activation="relu")(z)

output_gripper_state= Dense(1, activation="linear")(z)
output_joint_angles = Dense(7, activation="linear")(z)
output_gripper_pose = Dense(16, activation="linear")(z)
output_target_pose = Dense(16, activation="linear")(z)


model = Model(inputs=[input_gripper_state, input_joint_angles, input_depth_history], outputs=[output_gripper_state, output_joint_angles, output_gripper_pose, output_target_pose])


model.compile(optimizer='adam',
              loss={'output_gripper_state': 'sparse_categorical_crossentropy', 
                    'output_joint_angles': 'cosine_similarity',
                    'output_gripper_pose': 'mse',
                    'output_target_pose': 'mse',
                    }
)

print("\n---> Finished compiling the combined model.\n")

keras.utils.plot_model(model, "post_thesis_network_testing_model.png", show_shapes=True)

print("\n---> Finished compiling the combined model.\n")

# model.save('saved_model/compile_demo')
model.summary()

# print("\n---> Saved the combined model.\n")

# del model

# model = keras.models.load_model('saved_model/compile_demo')

# print("\n---> Loaded a saved model.\n")
