# BOPTEST-Gym

BOPTESTS-Gym is the [OpenAI-Gym](https://gym.openai.com/) environment for the [BOPTEST](https://github.com/ibpsa/project1-boptest) framework. This repository accommodates the BOPTEST API to the OpenAI-Gym convention in order to facilitate the implementation, assessment and benchmarking of reinforcement learning algorithms (RL) for their application to building energy management. RL algorithms from the [Stable-Baselines](https://github.com/hill-a/stable-baselines) repository are used to exemplify and test this framework. 

## Structure
- `boptestGymEnv.py` contains the core functionality of this Gym environment.
- `environment.yml` contains the dependencies required to run this software. 
- `/examples` contains prototype code for the interaction of RL algorithms with an emulator building model from BOPTEST. 
- `/testing` contains code for unit testing of this software. 

## Quick-Start
1) Create a conda environment from the `environment.yml` file provided (instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). 
2) Run a BOPTEST case with the building emulator model to be controlled (instructions [here](https://github.com/ibpsa/project1-boptest/blob/master/README.md)).  
3) Develop and test your own RL algorithms. See example here using the [Bestest hydronic case with a heat-pump](https://github.com/ibpsa/project1-boptest/tree/master/testcases/bestest_hydronic_heat_pump) and the [A2C algorithm](https://stable-baselines.readthedocs.io/en/master/modules/a2c.html) from Stable-Baselines: 

```python
from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, NormalizedObservationWrapper
from stable_baselines import A2C
from examples.test_and_plot import test_agent

# BOPTEST case address
url = 'http://127.0.0.1:5000'

# Instantite environment
env = BoptestGymEnvRewardWeightCost(url                   = url,
                                    actions               = ['oveHeaPumY_u'],
                                    observations          = {'reaTZon_y':(280.,310.)}, 
                                    random_start_time     = True,
                                    max_episode_length    = 24*3600,
                                    warmup_period         = 24*3600,
                                    Ts                    = 900)

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


