import random
from stable_baselines3 import DQN
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper

url = 'http://127.0.0.1'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)

def train_multiaction():
    '''Method to train a DQN agent with a multi-dimensional action environment. 

    '''

    env = BoptestGymEnv(
            url=url,
            testcase='singlezone_commercial_hydronic',
            actions=['oveTZonSet_u', 'oveTSupSet_u', 'oveCO2ZonSet_u'],
            observations={
                'time': (0, 604800),
                'reaTZon_y': (280., 310.),
                'TDryBul': (265, 303),
                'HDirNor': (0, 862),
                'InternalGainsRad[1]': (0, 219),
                'PriceElectricPowerHighlyDynamic': (-0.4, 0.4),
                'LowerSetp[1]': (280., 310.),
                'UpperSetp[1]': (280., 310.)
            },
            predictive_period=24*3600,
            regressive_period=6*3600,
            max_episode_length=24*3600,
            warmup_period=24*3600,
            step_period=3600,
            random_start_time=False,
            start_time=31*24*3600
        )

    # Normalize observations and discretize action space
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act=10)

    # Instantiate an RL agent
    model = DQN('MlpPolicy', env, verbose=1, gamma=0.99,
                learning_rate=5e-4, batch_size=24, seed=seed,
                buffer_size=365*24, learning_starts=24, train_freq=1)

    model.learn(total_timesteps=100)

    return env, model





