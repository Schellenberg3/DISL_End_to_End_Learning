# DISL End-to-End Learning

Welcome! This repository is a collection of tools for imitation training of neural networks on end-to-end
visuomotor control to complete tasks in a simulation environment. 



This was written as a part of my undergraduate honors thesis where I worked with Ohio State University's
[Design Innovation and Simulation Lab](https://disl.osu.edu/).  The thesis may be viewed 
[**here**]( http://hdl.handle.net/1811/92450). 

Note that the code has been updated and expanded since the thesis was published.

---

## Getting Started

### Requirements
This code relies on several well-developed libraries. These include: 

| Software | Version Used | Install Instructions | Use |
| --- | --- | --- | --- |
| CoppeliaSim | 4.1.0 | [Downloads page](https://www.coppeliarobotics.com/downloads) | Physics and robotics simulation environment |
| PyRep | 4.1.0.1 | [GitHub](https://github.com/stepjam/PyRep) | Python API for CoppeliaSim |
| RLBench | 1.0.10       | [GitHub](https://github.com/stepjam/RLBench) | Reinforcement learning toolbox |
| TensorFlow | 2.5.0 | [Installation guide ](https://www.tensorflow.org/install) | Creating and training neural networks |

TensorFlow is used for training and deploying the neural networks (NN).  CoppeliaSim provides a simulated 
environment in which we may collect training data or deploy a trained NN. It renders the environment and 
handles the physics. PyRep is a python API for CoppeliaSim. RLBench is a library that uses PyRep and 
CoppeliaSim to make it easy to design and randomly initialize a task environment for reinforcement learning.

Code has been tested in python 3.7 on Ubuntu 18.04 and 20.04.


### Installation

Each of these requirements provides some installation instructions which are linked in the 
[Requirements](#requirements) section. 

This section summarizes those and sets up the file structure to use this code. 

**This repo** and the others may be cloned with:

```shell
git clone https://github.com/Schellenberg3/DISL_End_to_End_Learning.git
git clone https://github.com/stepjam/PyRep.git
git clone https://github.com/Schellenberg3/RLBench.git
```

**CoppeliaSim** may be downloaded from the company's website 
[here](https://www.coppeliarobotics.com/downloads). The Edu version is used and is downloaded as a 
`.tar` file. To extract this to your home directory run:

 ```shell
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xc 
 ```

To verify that CoppeliaSim is working property you can run:

```shell
cd CoppeliaSim_Edu_V4_1_0_Ubuntu18_04
./coppeliaSim.sh
```

**PyRep** is installed after CoppeliaSim. Your *~/bashrc* file needs to be updated first. Add the following 
**(Note the 'EDIT/ME' in the first line)**

```shell
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Once you have saved this, you can pull PyRep from git and install it:

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

_It is important to note that a custom fork is used_. This contains a custom task for picking up a cup (DISL 
pick up blue cup). The [main repo](https://github.com/stepjam/RLBench) may be with minor edits.


**TensorFlow** may be installed by:

```shell
pip3 install tensorflow
```

We'll finish setting up the directories with:

```shell
cd ../DISL_End_to_End_Learning && mkdir data
```

Finally, to generate textures for domain randomization we run:
```shell
cd  utils && python perlin_noise.py
```

---

## Usage

The general process for using these programs is:

1. Create a new data set with `generate_episodes.py` 
   1. verify that a data set is properly created with `check_episodes.py`
2. Create a new NN and train it on a data set with `imitate.py`
3. Evaluate a trained NN against a data set with `evaluate.py` or `evaluate_on_dataset.py` 
4. Watch a trained NN perform against new data with `demonstrate.py` 

After imitation training, reinforcement learning could be performed. But there are no plans to introduce a
script to assist with this.

Each of these programs is written as a command line tool so minimal edits to the code should be needed. 
Things like what task to use or what camera point of view to use are automatically selected.

*This code was written for and has only been tested on tasks with one variation.* If you 
[add a new task](#how-to-create-a-custom-task) with variations then you may need to update the code.


### Generate Episodes

Running `python3 generate_episodes.py`  lets users create a data set of a single task.  Users select what 
task the dataset should be of and if domain randomization should be used.  The number of episodes is 
specified and a tag (testing, training, or misc) is assigned.

The program uses multithreading to generate episodes in parallel. Episodes are saved to the disc and may 
be tagged as **testing**, **training**, or **misc** (miscellaneous). Episodes with visual randomization
are saved in a separate directory from those without it.

The actual data for an episode is saved at a location similar to the following: `DISL_End_to_End_Learning/data/{tag}/{TaskName}/variation0/episodes/episode#`

In the file system the data structure is organized as:

```shell
DISL_End_to_End_Learning/
  └ data
    ├ misc
    |  ├ DislPickUpBlueCup
    |    └  variation0
    |       └ episodes
    |         ├ episode0   
    |         ├ episode1
    |         | . . . . 
    |         └ episode#
    ├ misc_randomized
    ├ testing
    ├ testing randomized
    ├ training
    └ training_randomized
```

This directory structure is generate by the code as you generate episodes, so you may only have a few 
of the directories listed under `data`.

The RLBench data is stored in the related `episode#` directory. Inside each episode's 
directory there is a folder with RGB, depth, and segmented images from the front, wrist, 
left and right cameras from each step in the episode.  Also included is `low_dim_obs.pkl` which 
stores information like joint positions and gripper state at each step.

_If episodes of a task with a tag already exist_, then this will add more episodes to reach the 
requested number.


### Check Episodes

Generating episodes may take a while, and the numbering of episodes is important, so if something 
interrupts generation then running `python3 check_episodes.py` will let a user select
a broken dataset and renumber it to bo continuous. 

This process also checks for episodes with abnormally high or low steps which may indicate issues with
that episode. It also makes sure that all data is present in each episode and flags those which may be
incomplete.

For a dataset with thousands of episodes, this process could take around 10 to 15 seconds.


### Imitate 

The `imitate.py` program is used for creating a new network and training it on one of the already
generated datasets. 

The structure of the network is defined in [`utils/networks.py`](utils/networks.py) but some high-level
options are available in `imitate.py`. These are:
   - Setting the number of input images 
   - Setting what point of view (wrist or front camera) to pull depth images from
   - Defining how many joint inputs there are 
   - Selecting if the joint inputs should be put through a DNN,
   - Assigning what **prediction mode (joint velocity or position)** the network should use

As part of the investigation into what LSTM options to use, both a stateful and stateless model are
saved. The networks are saved with the following structure: 

`DISL_End_to_End_Learning/imitation/j{number of Joints}g_v{number of images}_{point of view}_{number of training episodes}_by{number of epochs}_{prediction mode}`

On disk, the directories look something like:

```shell
DISL_End_to_End_Learning/
  └ networks
    └  imitation
       ├ j7g_v4_front_DislPickUpBlueCup_12000_by2
         ├ dataset_evaluations      # CSVs with details from evaluate_on_dataset.py
         ├ recordings               # Gifs of performances from evaluate_on_dataset.py
         ├ j7g_v4_front_DislPickUpBlueCup_12000_by2.h5   # the actual TensorFlow model
         ├ network.png              # image of the network's structure
         ├ network_info.pickle      # metainformation about the network
         └ train_performance.csv    # snapshot of training losses at 1% increments of training progress
       ├ j7g_v4_front_DislPickUpBlueCup_12000_by2_stateful    
         ├ . . . .               
         ├ j7g_v4_front_DislPickUpBlueCup_12000_by2_stateful.h5   # The stateless version of the network above
         └ . . . .   
       └ j7g_v1_wrist_ReachTarget_3000_by4
         ├ . . . .               
         ├ j7g_v1_wrist_ReachTarget_3000_by4.h5
         └ . . . .   
       └ j7g_v1_wrist_ReachTarget_3000_by4_stateful
         ├ . . . .               
         ├ j7g_v1_wrist_ReachTarget_3000_by4_stateful.h5
         └ . . . .   
```

This directory will be created as networks are trained.

Existing networks may be trained for additional epochs and will be saved in a new folder.

There is an option to evaluate the network immediately after it finishes training. This simply runs 
`evaluate.py` with the specified data.


### Evaluate

There are two options for evaluating a model against test episodes. 

`evaluate.py` is an implementation of TensorFlows's 
[`model.evaluate`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate) method. It is 
a good option for testing against a lot of episodes at once but only give losses as a per-episode 
average. **It is not recommended**.

Using `evaluate_on_dataset.py` is preferred for evaluating a network's performance. It recreates 
specified episodes from a dataset and lets the model perform for real. This give an accurate idea of 
the set-by-step performance and error of the network. Information on the episodes ground truth,
the network's predicted values, and the simulations actual values are recorded in the network's 
`dataset_evaluations` folder.


### Demonstrate

Running `python3 demostrate.py` will prompt the user to select a NN and how many demonstrations of the task 
they'd like to see.  CoppeliaSim will launch the task environment, so the NN can interact with the scene.

These are brand-new episodes that o not exist in any dataset and no information is recorded.


### Others

The `config.py` file contains the `EndToEndConfig` class which is used in all the other main programs. 
Its role is to manage the RLBench environment, TensorFlow session, and handle what information to
load from disk. It is worth previewing if you intend to use this code. 

The `view_episodes.py` is a convenient way to check out what episodes look like on disk, when loaded to
program memory, and when formatted for training. It is one way to test the code.

The `tests.py` provides several unit tests to verify the functionality of several core functions. 
Note that it may take a while to run because it generates new episodes during each run. 

The [`utils`](utils) directory abstracts some tools used throughout the code. 
[`utils/utils.py`](utils/utils.py) is especially of interest if you are curious how data is saved
or loaded to disk.

---

## Other Notes

### How to Create a Custom Task?

First, add the task in RLBench. Stephen James wrote two great guides on this [here](https://github.com/stepjam/RLBench/tree/master/tutorials).  When you've done that, simply add `from rlbench.tasks import YOUR_TASK` in `config.py` and then include your task in the `EndToEndConfig`'s dictionary `self.tasks`.



### How to Create a Custom Neural Network?

The neural network structure is defined in several functions inside the
[`utils/netowrks.py`](utils/networks.py) NetworkBuilder class. Simply editing these will make it easy to 
adjust your network's structure.


### How to use a Robot/Gripper other than Panda?

If a robot can be [added to PyRep](https://github.com/stepjam/PyRep/blob/master/tutorials/adding_robots.md) and [added to RLBench](https://github.com/stepjam/RLBench/issues/111) then it should be possible to add it to this code. However, this has not been tested. At a minimum, the following changes would need to be made:

