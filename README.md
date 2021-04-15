# DISL End-to-End Learning

Welcome! This repository is a collection of tools for imitation training of neural networks on end-to-end visuomotor control to complete tasks in a simulation environment. 



This was written as an part of my undergraduate honors thesis where I worked with Ohio State University's [Design Innovation and Simulation Lab](https://disl.osu.edu/).  The thesis may be viewed **here**.



## Getting Started

### Requirements
This code relies on several well developed libraries. These include: 

| Software | Version Used | Install Instructions | Use |
| --- | --- | --- | --- |
| CoppeliaSim | 4.1.0 | [Downloads page](https://www.coppeliarobotics.com/downloads) | Physics and robotics simulation environment |
| PyRep | 4.1.0.1 | [GitHub](https://github.com/stepjam/PyRep) | Python API for CoppeliaSim |
| RLBench | 1.0.10       | [GitHub](https://github.com/stepjam/RLBench) | Reinforcement learning toolbox |
| TensorFlow | 2.5.0 | [Instalation guide ](https://www.tensorflow.org/install) | Creating and training neural networks |

TensorFlow is used for training and deploying the neural networks (NN).  CoppeliaSim provides a simulated environment in which we may collect training data or deploy a trained NN. It renders the environment and handles the physics. PyRep is a python API for CoppeliaSim. RLBench is a library that uses PyRep and CoppeliaSim to make it easy to design and randomly initialize a task environment for reinforcement learning.

Code was written in python 3.6.9 and tested on Ubuntu 18.04. 



### Installation

Each of these requirements provides some installation instructions which are linked in the [Requierments](#requierments) section. 

This section summarizes those and sets up the file structure to use this code. 

**This repo** and the others may be cloned with:

```shell
git clone https://github.com/Schellenberg3/DISL_End_to_End_Learning.git
git clone https://github.com/stepjam/PyRep.git
git clone https://github.com/Schellenberg3/RLBench.git
```

**CoppeliaSim** may be downloaded from the company's website [here](https://www.coppeliarobotics.com/downloads). The Edu version is used and is downloaded as a `.tar` file. To extract this to your home directory run:

 ```shell
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xc 
 ```

 To verify that CoppeliaSim is working property you can run:

```shell
cd CoppeliaSim_Edu_V4_1_0_Ubuntu18_04
./coppeliaSim.sh
```

**PyRep** is installed after CoppeliaSim. Your *~/bashrc* file needs to be updated first. Add the following **(Note the 'EDIT/ME' in the first line)**

```shell
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Once you have save this, you can pull PyRep from git and install it:

```shell
source ~/.bashrc
cd PyRep
pip3 install -r requirements.txt
pip3 install .
cd ..
```

**RLBench** may be cloned and installed by:

```shell
cd RLBench
pip3 install -r requirements.txt
pip3 install .
```

It is important to note that a custom fork is used. This contains a custom task for picking up a cup (DISL pick up blue cup). The [main repo](https://github.com/stepjam/RLBench) may be used if all refences to this class are removed from the code in these files.


**TensorFlow** may be installed by:

```shell
pip3 install tensorflow
```

**Finally** we'll finish setting up the directories with:

```shell
cd ../DISL_End_to_End_Learning && mkdir data

```



## How to Use

The process for using these programs is:

1. Create a new data set with `generate_episodes.py` 
   1. verify that a data set is properly created with `check_episodes.py`
2. Create a new NN and train it on a data set with `imitate.py`
3. Evaluate a trained NN against a data set with `evaluate.py` 
4. Watch a trained NN perform with `demonstrate.py` 

Another stage of reinforcement learning is planed with  `reinforce.py` but this has not been implemented yet.

Each of these programs is written as a command line tool so minimal edits to the code should be needed. Things like what task to use or what camera point of view to use are automatically selected.

*This code was written for and has only been tested on tasks with one variation.* If you [add a new task](#How-to-Create-a-Custom-Task) with variations then you may need to update the code.



#### Generate Episodes

Running `python3 generate_episodes.py`  lets users create a data set of a single task.  Users select what task the dataset should be of and if domain randomization should be used.  The number of episodes is specified and a tag (testing, training, or misc) is assigned.

The program uses multithreading to generate episodes in parallel. Episodes saved in the following structure:

```shell
data/{tag}/{TaskName}/variation0/episodes/episode#
```

Where **#** represents the episode number. Inside each episodes directory there is a folder with RGB, depth, and segmented images from the from the front, wrist, left and right cameras from each step in the episode.  Also included is `low_dim_obs.pkl` which stores information like joint positions and gripper state at each step.

If episodes of a task with a tag already exist, then this will add more episodes to reach the requested number.



#### Check Episodes

Generating episodes may take a while and the numbering of episodes is important. If something interrupts generation then running `python3 check_episodes.py` will let a user select a broken dataset and renumber it so it is continuous. 

It also checks the number of steps in each episode to look for outliers with fewer than or more than the average number of steps.



#### Imitate 

Once a dataset is created `python3 imitate.py ` will launch a tool for training a neural network.  Users select a dataset to test on,  a dataset to train on, how many testing and training episodes to use, and how many epochs. The task is inferred from the directories name. Users may specify if RGBD images from the wrist or the front point of view are used.

Finally, the type of network is selected. There are four options for this:

| Structure             | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| Position Vision       | NN with current position and RGBD image as input             |
| Position Vision-4     | NN with current position and current + previous three RGBD images as input |
| RNN Position Vision   | NN with current position and RGBD image as input and a LSTM layer |
| RNN Position Vision-4 | NN with current position and current + previous three RGBD images as input and a LSTM layer |

A [custom NN structure](#how-to-create-a-custom-neural-network) could easily be added.

NNs are saved in the following structure:

```shell
# non domain randomized data set:
networks/imitation/{NN strucure}_{CNN-Structure}_{Task Name}_{Training Amount}_by{Epochs}

# domain randomized data set: 
networks/imitation/{NN strucure}_{CNN-Structure}_{Task Name}_{Training Amount}_by{Epochs}_rand
```

In the NN's directory `model_summary.txt` records the results from the testing episodes.  Snapshots of the models MSE over the course of training are saved in `train_performance.csv`    

An already trained network may be loaded in for additional training epochs.


#### Evaluate

Running `python3 evaluate.py` will prompt the user to select a trained network and a data set.  The network is evaluated against a specified number of episodes from that data set.  This information is appended to `model_summary.txt` 



#### Demonstrate

Running `python3 demostrate.py` will prompt the user to select a NN and how many demonstrations of the task they'd like to see.  CoppeliaSim will launch the task environment and the NN can interact with the scene.



#### Reinforce 

This has yet to be written and will not be part of the undergraduate thesis.  This is included as a next step for this project.



#### Config

The `config.py` contains the `EndToEndConfig` class which is used in all the other main programs. In it the variables `self.custom_networks` and `self.tasks` are dictionaries that control what networks and tasks the code uses, respectively.  These can be expanded to include more options.



## Other Notes

### How to Create a Custom Task?

First, add the task in RLBench. Stephen James wrote two great guides on this [here](https://github.com/stepjam/RLBench/tree/master/tutorials).  When you've done that, simply add `from rlbench.tasks import YOUR_TASK` in `config.py` and then include your task in the `EndToEndConfig`'s dictionary `self.tasks`.



### How to Create a Custom Neural Network?

In `utils/networks.py` you can create a function `CUSTOM_NN` that uses TensorFlow to construct a NN model. The function should return the network without compiling it - this is done in `imitate.py`.

In `config.py` add `from utils.networks import CUSTOM_NN`  and add this function to `EndToEndConfig`'s dictionary `self.custom_netowrks`.  

In the dictionary you'll also include what function to use to split each step of an episode into the position input, image input, and ground truth label. 



### How to use a Robot/Gripper other than Panda?

If a robot can be [added to PyRep](https://github.com/stepjam/PyRep/blob/master/tutorials/adding_robots.md) and [added to RLBench](https://github.com/stepjam/RLBench/issues/111) then it should be possible to add it to this code. However, this has not been tested. At a minimum, the following changes would need to be made:

* The number of position inputs in `utils/networks` would need to equal the robots joints plus one (for gripper state)

