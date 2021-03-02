# DISL End-to-End Learning

Welcome! This repository is a collection of tools for imitation  and reinforcement learning robotics tasks in a simulated environment.

## Getting Started

### Installation
To running these files require that some some dependencies be installed. These are:

| Software | Install Instructions | Use |
| --- | --- | --- |
| CoppeliaSim | [downloads page](https://www.coppeliarobotics.com/downloads) | Physics Simulator |
| PyRep | [GitHub](https://github.com/stepjam/PyRep) | Python API for CoppeliaSim |
| RLBench | [GitHub](https://github.com/stepjam/RLBench) | Reinforcement learning toolbox |
| TensorFlow | [install page](https://www.tensorflow.org/install) | Creating and training neural networks |

In short, RLBench lets us design tasks, generate training data, and provides rewards for reinforcement learning.  To do this it requires PyRep to communicate with the physics simulations done in CoppeliaSim. TensorFlow handles everything for the neural networks we create.

Linux in required for this.  All code was written and tested on Ubuntu 18.04.

## Contents
 
The various files and directories in the project attempt to break the complex task of data generation and training into discrete steps. There are specific files for generating a data set, to creating a neural network, to training it, and running it. 

#### Workflow files

These are the main files that should be executed in generating a data set and training a neural network.

| File | Description |
| --- | --- |
| `single_task_data_generator.py` | Creates a specified number of demonstrations for a single RLBench task.  It can be used to add new episodes to an existing data set. |
| `data_helper.py` | Re-numbers episodes to ensure a continuous data set, provides information about the number of steps in each episode, and point out any episodes that may be incomplete. |
| `imitation_learner.py`| Loads a network from `disl_networks.py` to train on an existing data set and validate against an existing testing data set. Saves the models in the `imitation_trained/`. |
| `reinforcement_learner.py` | Currently a placeholder. This would load a model from `imitation_trained/` and continue training it with reinforcement learning before saving the final model in `reinforcement_trained/`. |
| `demonstrator.py` | Loads a model from either directory of trained models and evaluates its performance on a task. |

####Support files

These programs are extra tools that provide some utility the main workflow or contain objects called by them. 

| File | Description |
| --- | --- |
| `data_validation.py` | Creates a small data set and reloads it to validate that each episode can be saved and reloaded. Provides some visuals showing what the networks training data looks like. |
| `disl_utils.py` | Provides common code for saving a demonstration and loading a saved demonstration. It also splits loaded data to get the inputs and label at each time step of the episode in `imitation_learner.py`. |
| `disl_networks.py` | Contains functions that return the various TensorFlow networks may be used in `imitation_learner.py`. |

#### Directories

| Folder | Description |
| --- | --- |
| `archive/` | Holds old code which will likely be deleted in the future. |
| `data/` | Holds all the data sets used in the project. **Due to its size this is not tracked on git.**  Use `single_task_data_generator.py` to populate this yourself |
| `network_info/` | Contains the output of `disl_networks.py` which is a description and images for the network structures |
| `trained_networks/` | Has two sub-directories `imitation/` and `reinforcement/` which hold networks trained via their respective methods. |

