from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench import DomainRandomizationEnvironment
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
import numpy as np
# Custom writen imports
from disl_utils import save_demos
from disl_utils import format_data
from disl_utils import get_order
from disl_utils import load_data
from disl_utils import split_data


def get_model():
    # model is inspired by Herman et al., 2020 and James et al., 2017
    # the first branch operates on the first input, the joint positions
    inputA = Input(shape=8)
    x = Dense(64, activation="tanh")(inputA)
    x = Dense(64, activation="tanh")(x)
    x = Dense(64, activation="tanh")(x)
    # x = Dense(1, activation="tanh")(x)
    x = Flatten()(x)
    model_x = Model(inputs=inputA, outputs=x)

    # the second branch operates on the second input, the camera image
    # this input is the 128x128 image with 3 channels
    inputB = Input(shape=(128, 128, 4))
    filters = (32, 64, 32)

    size = [(8, 8), (4, 4), (2, 2)]
    stride = [4, 2, 1]
    chanDim = -1

    y = inputB
    for (i, f) in enumerate(filters):
        # CONV => RELU => BN => POOL
        y = Conv2D(f, size[i], strides=stride[i], padding="same")(y)
        y = Activation("relu")(y)
        y = BatchNormalization(axis=chanDim)(y)
        y = MaxPooling2D(pool_size=(2, 2))(y)

    y = Flatten()(y)
    y = Dense(16)(y)
    y = Activation("relu")(y)
    y = BatchNormalization(axis=chanDim)(y)
    y = Dropout(0.5)(y)
    y = Dense(512)(y)
    y = Activation("relu")(y)

    model_y = Model(inputB, y)

    combined = Concatenate()([model_x.output, model_y.output])

    z = Dense(128, activation="relu")(combined)
    z = Reshape((1, 128))(z)
    z = LSTM(64, return_sequences=True, input_shape=(1, 1, 128))(z)
    z = Dense(128, activation="relu")(z)
    z = Dense(128, activation="relu")(z)
    z = Dense(8, activation="linear")(z)

    _model = Model(inputs=[model_x.input, model_y.input], outputs=z)

    return _model


no_model = True
live_demos = False
split = 0.8

DATASET = '' if live_demos else 'train_demos'

obs_config = ObservationConfig()
obs_config.set_all(True)

''' for example domain randomization '''
rand_config = VisualRandomizationConfig(
    image_directory='../tests/unit/assets/textures')

action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
env = Environment(action_mode, DATASET, obs_config, False)

'''
env = DomainRandomizationEnvironment(
    action_mode, obs_config=obs_config, headless=False,
    randomize_every=RandomizeEvery.EPISODE, frequency=1,
    visual_randomization_config=rand_config
)
'''

env.launch()

task = env.get_task(ReachTarget)


if live_demos:
    num_live_demos = 200
    demos = task.get_demos(num_live_demos, live_demos=live_demos)  # -> List[List[Observation]]

    save_path = 'train_demos_random/reach_target/'
    save_demos(demos, save_path)

    print('---> Exiting program.  For now generate demos with live demos then rerun'
          'and load the saved demos')

    env.shutdown()

    quit()


if no_model:

    # todo: do we still need this?
    '''
    The train/test data structure is a scary mixture of lists and arrays... but it is needed for Tensorflow

    test_data           -> list of each test demo
    test_data[L]        -> list of the robot state for each step in demo L
    test_data[L][M]     -> ndarry of robot pos and gripper state at step M of demo L
    test_data[L][M][N]  -> float or int of state N at step M of demo L

    *_image, *_label, and *_data all work like this 
    '''

    model = get_model()
    model.compile(optimizer='adam',
                  loss="mean_squared_error",
                  metrics=['mse'])

    '''----------TRAINING----------'''

    print("\n---> Finished compiling the model.\n")

    train_path = 'train_demos_random/reach_target/variation0/episodes'
    train_episodes = 200
    epochs = 10
    train_order = get_order(train_episodes, epochs)
    count = 0

    for train in train_order:
        demo = load_data(train_path, train, obs_config)
        demo = format_data(demo)
        train_data, train_images, train_label = split_data(demo)

        model.fit(x=[np.asarray(train_data),
                     np.asarray(train_images)],
                  y=np.asarray(train_label),
                  shuffle=False,
                  epochs=1)

        print(f'---> EPOCH {int(count/train_episodes) + 1 }/{epochs}:  '
              f'Fit model for training demo #{train} with {len(train_data)} data points. '
              f'This is step {count + 1} of {train_episodes * epochs}.'
              f'\n')

        count += 1

    print("\n---> Finished training model.\n\n")

    model.save('saved_model/reach_pos_vis_10E_random')

    print("\n\n---> Saved the model.\n\n")

else:
    model = tf.keras.models.load_model('saved_model/demo_1_testing')
    print("---> Finished loading a saved the model.\n", model.summary(), "\n")

'''----------TESTING----------'''

test_path = 'test_demos_random/reach_target/variation0/episodes'
test_episodes = 20
test_order = get_order(test_episodes)
count = 0
tot_acc = 0

for test in test_order:
    demo = load_data(test_path, test, obs_config)
    demo = format_data(demo)
    test_data, test_images, test_label = split_data(demo)

    loss, acc = model.evaluate(x=[np.asarray(test_data),
                                  np.asarray(test_images)],
                               y=np.asarray(test_label))

    print(f"---> Evaluated test demo {count + 1} of {test_episodes}."
          f" Found accuracy of {acc * 100}%\n")

    tot_acc += acc
    count += 1

print(f"\n---> Finished testing model on domain-randomized instances. "
      f"Total accuracy of: {100 * tot_acc/count}% \n\n")


test_path = 'test_demos/reach_target/variation0/episodes'
test_episodes = 20
test_order = get_order(test_episodes)
count = 0
tot_acc = 0

for test in test_order:
    demo = load_data(test_path, test, obs_config)
    demo = format_data(demo)
    test_data, test_images, test_label = split_data(demo)

    loss, acc = model.evaluate(x=[np.asarray(test_data),
                                  np.asarray(test_images)],
                               y=np.asarray(test_label))

    print(f"---> Evaluated test demo {count + 1} of {test_episodes}."
          f" Found accuracy of {acc * 100}%\n")

    tot_acc += acc
    count += 1

print(f"\n---> Finished testing model. Total accuracy of: {100 * tot_acc/count}% \n\n")

evaluation_steps = 200
episode_length = 40
obs = None
for i in range(evaluation_steps):
    if i % episode_length == 0:
        descriptions, obs = task.reset()
        print(f"---> Task reset: on episode {int(1+(i/episode_length))} of {int(evaluation_steps / episode_length)}")

    image = np.expand_dims(np.dstack((obs.front_rgb, obs.front_depth)), 0)
    state = np.expand_dims(np.append(obs.joint_positions, obs.gripper_open), 0)
    action = model.predict(x=[state, image])
    # action[0][7] = round(action[0][7])  # round to either open or closed
    # print(f'---> Action at step {i+1} is {action}')
    obs, reward, terminate = task.step(action.flatten())
    # print(f'---> reward = {reward}    terminate = {terminate}')

print('Done!!')
env.shutdown()


