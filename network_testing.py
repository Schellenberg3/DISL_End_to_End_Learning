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

# model is inspired by Herman et al., 2020 and James et al., 2017
# the first branch operates on the first input, the joint positions

inputA = Input(shape=8)
x = Dense(64, activation="tanh")(inputA)
x = Dense(64, activation="tanh")(x)
x = Dense(64, activation="tanh")(x)
x = Flatten()(x)
model_x = Model(inputs=inputA, outputs=x)

print("\n---> Finished defining the position model.\n")

# the second branch operates on the second input, the camera image
# this input is the 128x128 image with 3 channels
inputB = Input(shape=(128, 128, 4))

# Filters are an integer representing the dimensionality of the output space
filters = (32, 64, 32)

# The size of the kernel window applied for each filter
size = [(8, 8), (4, 4), (2, 2)]

# Stride defines how many pixels we move before the next convolution is applied
#    Practically, this means the dimensions of the output will be the input / stride
#    e.g. an 8x8 image after a stride of 2 will be 4x4
#    Stride may also be a tuple for the height and length dimensions
#
#    It seems that the stride is usually half the kernel window's dimensions
stride = [4, 2, 1]

chanDim = -1

y = inputB
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
    y = Conv2D(f, size[i], strides=stride[i], padding="same", activation="relu")(y)
    y = BatchNormalization(axis=chanDim)(y)
    y = Activation("relu")(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)

y = Flatten()(y)
y = Dense(16)(y)
y = Activation("relu")(y)
y = BatchNormalization(axis=chanDim)(y)
# Drop out randomly sets some of the inputs from the prior layer to zero
#    This makes the network more robust to inputs by simulating s sparse network
#    Doing so also approximates training different structures
#    and prevents the network from relying on just a few nodes, all must learn to be useful
#    Nodes are set to zero at the probability defined in the call (e.g. Dropout(0.8) -> 80% chance of 0)
#
#    This layer only drops nodes during training.  When evaluating the dropout rate is 0%
y = Dropout(0.5)(y)
y = Dense(512)(y)
y = Activation("relu")(y)

model_y = Model(inputB, y)

print("\n---> Finished defining the vision model.\n")

combined = Concatenate()([model_x.output, model_y.output])


z = Dense(128, activation="relu")(combined)
z = Reshape((1, 128))(z)
z = LSTM(64, return_sequences=True, input_shape=(1, 1, 128))(z)
z = Dense(128, activation="relu")(z)
z = Dense(128, activation="relu")(z)
z = Dense(8, activation="linear")(z)

model = Model(inputs=[model_x.input, model_y.input], outputs=z)
model.compile(optimizer='adam',
              loss="mean_squared_error",
              metrics=['accuracy'])

print("\n---> Finished compiling the combined model.\n")

keras.utils.plot_model(model, "model.png", show_shapes=True)

print("\n---> Finished compiling the combined model.\n")

model.save('saved_model/compile_demo')
model.summary()

print("\n---> Saved the combined model.\n")

del model

model = keras.models.load_model('saved_model/compile_demo')

print("\n---> Loaded a saved model.\n")

