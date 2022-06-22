# BOPTEST-Gym-service

BOPTESTS-Gym is the [OpenAI-Gym](https://gym.openai.com/) environment for the [BOPTEST](https://github.com/ibpsa/project1-boptest) framework. This repository accommodates the BOPTEST API to the OpenAI-Gym convention in order to facilitate the implementation, assessment and benchmarking of reinforcement learning (RL) algorithms for their application in building energy management. RL algorithms from the [Stable-Baselines 3](https://github.com/DLR-RM/stable-baselines3) repository are used to exemplify and test this framework. 

This is the service version of BOPTEST-Gym, meaning that it has been adapted to use BOPTEST test cases that can be run in a server instead of just locally. 

The environment is described in [this paper](https://www.researchgate.net/publication/354386346_An_OpenAI-Gym_environment_for_the_Building_Optimization_Testing_BOPTEST_framework). 

## Structure
- `boptestGymEnv.py` contains the core functionality of this Gym environment.
- `environment.yml` contains the dependencies required to run this software. 
- `/examples` contains prototype code for the interaction of RL algorithms with an emulator building model from BOPTEST. 
- `/testing` contains code for unit testing of this software. 

## Quick-Start
1) Create a conda environment from the `environment.yml` file provided (instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). 
2) Run a BOPTEST case with the building emulator model to be controlled (instructions [here](https://github.com/ibpsa/project1-boptest/blob/master/README.md)).  
3) Develop and test your own RL algorithms. See example below using the [Bestest hydronic case with a heat-pump](https://github.com/ibpsa/project1-boptest/tree/master/testcases/bestest_hydronic_heat_pump) and the [A2C algorithm](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html) from Stable-Baselines: 

```python
from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper
from stable_baselines3 import A2C
from examples.test_and_plot import test_agent

# BOPTEST case address
url = 'http://127.0.0.1'

# Instantite environment
env = BoptestGymEnv(url                   = url,
                    testcase              = 'bestest_hydronic_heat_pump',
                    actions               = ['oveHeaPumY_u'],
                    observations          = {'reaTZon_y':(280.,310.)}, 
                    random_start_time     = True,
                    max_episode_length    = 24*3600,
                    warmup_period         = 24*3600,
                    step_period           = 900)

# Add wrappers to normalize state and action spaces (Optional)
env = NormalizedObservationWrapper(env)
env = NormalizedActionWrapper(env)  

# Instantiate and train an RL algorithm
model = A2C('MlpPolicy', env)
model.learn(total_timesteps=int(1e5))

# Test trained agent
observations, actions, rewards, kpis = test_agent(env, model, 
                                                  start_time=0, 
                                                  episode_length=14*24*3600,
                                                  warmup_period=24*3600,
                                                  plot=True)

```

## Citing the project

Please use the following reference if you used this repository for your research.

```
@inproceedings{boptestgym2021,
	author = {Javier Arroyo and Carlo Manna and Fred Spiessens and Lieve Helsen},
	title = {{An OpenAI-Gym environment for the Building Optimization Testing (BOPTEST) framework}},
	year = {2021},
	month = {September},
	booktitle = {Proceedings of the 17th IBPSA Conference},
	address = {Bruges, Belgium},
}

```



