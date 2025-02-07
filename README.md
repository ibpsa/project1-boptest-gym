# BOPTEST-Gym

BOPTESTS-Gym is the [Gymnasium](https://gymnasium.farama.org/index.html) environment of the [BOPTEST](https://github.com/ibpsa/project1-boptest) framework. This repository accommodates the BOPTEST API to the Gymnasium standard in order to facilitate the implementation, assessment and benchmarking of reinforcement learning (RL) algorithms for their application in building energy management. RL algorithms from the [Stable-Baselines 3](https://github.com/DLR-RM/stable-baselines3) repository are used to exemplify and test this framework. 

The environment is described in [this paper](https://www.researchgate.net/publication/354386346_An_OpenAI-Gym_environment_for_the_Building_Optimization_Testing_BOPTEST_framework). 

## Structure
- `boptestGymEnv.py` contains the core functionality of this Gymnasium environment.
- `environment.yml` contains the dependencies required to run this software. 
- `/examples` contains prototype code for the interaction of RL algorithms with an emulator building model from BOPTEST. 
- `/testing` contains code for testing this software. 

## Quick-Start

1) Create an environment from the `environment.yml` file provided (instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). You can also see our Dockerfile in [testing/Dockerfile](testing/Dockerfile) that we use to define our testing environment. 
2) Run the example below that uses the [Bestest hydronic case with a heat-pump](https://github.com/ibpsa/project1-boptest/tree/master/testcases/bestest_hydronic_heat_pump) and the [DQN algorithm](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) from Stable-Baselines: 

```python
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from stable_baselines3 import DQN

# url for the BOPTEST service. 
url = 'https://api.boptest.net' 

# Decide the state-action space of your test case
env = BoptestGymEnv(
        url                  = url,
        testcase             = 'bestest_hydronic_heat_pump',
        actions              = ['oveHeaPumY_u'],
        observations         = {'time':(0,604800),
                                'reaTZon_y':(280.,310.),
                                'TDryBul':(265,303),
                                'HDirNor':(0,862),
                                'InternalGainsRad[1]':(0,219),
                                'PriceElectricPowerHighlyDynamic':(-0.4,0.4),
                                'LowerSetp[1]':(280.,310.),
                                'UpperSetp[1]':(280.,310.)}, 
        predictive_period    = 24*3600, 
        regressive_period    = 6*3600, 
        random_start_time    = True,
        max_episode_length   = 24*3600,
        warmup_period        = 24*3600,
        step_period          = 3600)

# Normalize observations and discretize action space
env = NormalizedObservationWrapper(env)
env = DiscretizedActionWrapper(env,n_bins_act=10)

# Instantiate an RL agent
model = DQN('MlpPolicy', env, verbose=1, gamma=0.99,
            learning_rate=5e-4, batch_size=24, 
            buffer_size=365*24, learning_starts=24, train_freq=1)

# Main training loop
model.learn(total_timesteps=10)

# Loop for one episode of experience (one day)
done = False
obs, _ = env.reset()
while not done:
  action, _ = model.predict(obs, deterministic=True) 
  obs,reward,terminated,truncated,info = env.step(action)
  done = (terminated or truncated)

# Obtain KPIs for evaluation
env.get_kpis()

```

In [this tutorial](https://github.com/ibpsa/project1-boptest-gym/blob/master/docs/tutorials/CCAI_Summer_School_2022/Building_Control_with_RL_using_BOPTEST.ipynb) you can find more details on how to use BOPTEST-Gym and on RL applied to buildings in general. 

### Note 1: on running BOPTEST in the server vs. locally
The previous example interacts with BOPTEST in a server at `https://api.boptest.net` which is readily available anytime. Interacting with BOPTEST from this server requires less configuration effort but is slower because of the communication overhead between the agent and the test case running in the cloud. Use this approach when you want to quickly check out the functionality of this repository. 

If you prioritize speed (which is usually the case when training RL agents), running BOPTEST locally is substantially faster. 
You can do so by downloading the BOPTEST repository and running:
```bash
docker compose up web worker provision

```

Further details in the [BOPTEST GitHub page](https://github.com/ibpsa/project1-boptest/blob/master/README.md#quick-start-to-deploy-and-use-boptest-on-a-local-computer). 

Then you only need to change the `url` to point to your local BOPTEST service deployment instead of the remote server (`url = 'http://127.0.0.1').

### Note 2: on running BOPTEST locally in a vectorized environment

BOPTEST allows the deployment of multiple test case instances using Docker Compose. 
Running a vectorized environment enables the deployment of as many BoptestGymEnv instances as cores you have available for the agent to learn from all of them in parallel. See [here](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) for more information, we specifically use [`SubprocVecEnv`](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv). This can substantially speed up the training process. 

To do so, deploy BOPTEST with multiple workers to spin multiple test cases. See the example below that prepares BOPTEST to spin two test cases.

```bash
docker compose up --scale worker=2 web worker provision
```

Then you can train an RL agent with parallel learning with the vectorized BOPTEST-gym environment. See [`/examples/run_vectorized.py`](https://github.com/ibpsa/project1-boptest-gym/blob/master/examples/run_vectorized.py) for an example on how to do so. 

## Versioning and main dependencies

Current BOPTEST-Gym version is `v0.7.0` which is compatible with BOPTEST `v0.7.0` 
(BOPTEST-Gym version should always be even with the BOPTEST version used). 
The framework has been tested with `gymnasium==0.28.1` and `stable-baselines3==2.0.0`.
You can see [testing/Dockerfile](testing/Dockerfile) for a full description of the testing environment. 

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