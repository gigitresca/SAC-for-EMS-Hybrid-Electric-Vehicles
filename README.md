# SAC-for-EMS-Hybrid-Electric-Vehicles
Official implementation of the paper [Development of a Soft-Actor Critic reinforcement Learning Algorithm for the Energy Management of Hybrid Electric Vehicle](https://www.sae.org/publications/technical-papers/content/2024-37-0011/)
presented at CO2 Reduction for Transportation Systems Conference 2024, held in Turin.

<img align="center" src="figures/tools.png" width="1500"/></td> <br/>
<img align="center" src="figures/training_results.png" width="1500"/></td> <br/>

This repository provides the code and methodology for setting up and training a Soft-Actor Critic (SAC) reinforcement learning algorithm to optimize the energy management of a Hybrid Electric Vehicle (HEV). The implementation is built using Python and leverages the OpenAI Gym environment to simulate the HEV system and the Stable Baselines3 library to set the SAC agent.

The repository includes detailed explanations and modular code for:

- HEV backward kinematic model compliant with the OpenAI Gym framework
- SAC algorithm implementation using Stable Baselines3 library
- Training and testing procedures
- Methodology to postprocess the output of the simulation

The code is designed to be modular, allowing for 
- Easy integration of new Stable Baselines3 agents
- Easy customization of the vehicle model environment
- Easy integration of different reward functions

## Prerequisites
- The repo is tested on Python 3.12 version
- The environments are built on Gymnasium 1.0.0 library
- The SAC agent was implemented using StableBaselines3 2.5.0
- All the additional packages are reported in the file `requirements.txt`

## Contents
The repository is structured in the following way:

`SAC-for-EMS-Hybrid-Electric-Vehicles
`  
`├── test.py`: test pre-trained RL agents <br>
`├── train.py`: main file for training RL agents <br>
`├── src/`<br>
`    ├── envs/`<br>
`        ├── vehicle_objects/`: classes for each physical component of the vehicle, for the reward and states normalization<br>
`        ├── VehicleBackKinEnv.py`: backward kinematic model environment class, wrapped from gym.Env <br>
`    ├── utilities/`: helper functions<br>
`├── data/`<br>
`    ├── driving_cycles.xlsx`: Excel file with training/testing driving cycles<br>
`    ├── vehicle_data.xlsx`: Excel file with vehicle parameters for the case study<br>
`    ├── driving_cycles.pkl`: pkl file with post-processed driving cycles from driving_cycles.xlsx<br>

## Configuration files
The code uses an argument parser (argparse) to set parameters directly from the command line.
To run the script with custom parameters, use the following syntax:

```
python script.py --param1 value1 --param2 value2 --optional_flag
```

The available parameters are the following:
```
        agent_name            SAC agent name (default: "date_of_today_SAC")
        actor_hidden_nodes    number of nodes for the hidden layers of the policy network (default: 256)
        critic_hidden_nodes   number of nodes for the hidden layers of the critic network (default: 256)
        lr                    learning rate for the policy nets update (default: 0.0003)
        batch                 batch size for the nets training (default: 256)
        tau                   soft update coefficient for the target critic (default: 0.005)
        gamma                 discount factor (default: 0.99)
        episodes              episodes to run during the training phase (default: 500)
```

## Usage

### Training

To train an agent, follow these steps:
- Run the `train.py` script with the desired parameters as specified in the previous section.
- After training, the model checkpoints are saved in the directory `result/saved_agents/.`
- After training, the log files are saved in the directory `result/logs/.`

### Testing

To test a pre-trained agent, follow these steps:
- Run the `test.py` script.
- Choose a pre-trained agent with the parameter `agent_name`.
- After the simulation, plots displaying key vehicle quantities (e.g., SoC, ICE power) will be generated.

### Change additional agent parameters

To change parameters that are not specified as arguments, you must modify the train.py script directly.
The default parameters of the agent are specified in line 64, where the SAC class is called.

### Change vehicle parameters

To change vehicle parameters:
- Update the `data/vehicle_data.xlsx` file.

To change or add driving cycles:
- Delete the file `data/driving_cycles.pkl`
- Update the `data/driving_cycles.xlsx` file.
- Regenerate the `data/driving_cycles.pkl` file.

### Add new agents

To add new agents, you just need to be compliant with StableBaselines and its guidelines to add different RL agents

### Add new reward
To add a customized reward:
- Implement a custom reward class in the `src/envs/VehicleBackwardKin.py` script.
- Update line 81 of `src/envs/VehicleBackwardKin.py` script, in order to define the custom reward
    - The three rewards presented in the paper can be selected here in this repository by specifying the string "dense", "sparse", "soc_bc" in the reward definition line.
 
## Conclusion

This repository provides a comprehensive framework for training and testing reinforcement learning agents to optimize the energy management of a gymnasium-based plug-in Hybrid Electric Vehicle. With ready-to-use SAC implementations, a modular environment setup, and clear instructions for extending agents, rewards, and vehicle parameters, the codebase is designed for flexibility and ease of customization.
Thank you for your interest in this project! We welcome contributions from the community to contribute to the development of this project.
Feel free to open issues, suggest improvements, or submit pull requests to help expand and enhance this repository. 🚀
