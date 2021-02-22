The goal of demo1.py is demonstrating the imitation learning method for a simple task.

The panda arm and gripper are used to complete the reaching task.

Our network is a mixture of a CNN for the image and Deep NN for the state.

These models will feed into one LSTM network to predict the next state of the robot.

100 demonstration samples are used. No reinforcement learning is performed after.

A second demo will expand on this to incorporate multiple images in the input.ee_

