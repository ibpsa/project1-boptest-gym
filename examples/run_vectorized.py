import os
import sys
import yaml
import torch
import random

from testing import utilities
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.logger import configure
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper

def generate_urls_from_yml(boptest_root_dir):
    '''Method that returns as many urls for BOPTEST-Gym environments 
    as those specified at the BOPTEST `docker-compose.yml` file. 
    It assumes that `generateDockerComposeYml.py` has been called first. 

    Parameters
    ----------
    boptest_root_dir: str
        String with directory to BOPTEST where the `docker-compose.yml` 
        file should be located. 

    Returns
    -------
    urls: list
        List of urls where BOPTEST test cases will be allocated. 

    '''
    docker_compose_loc = os.path.join(boptest_root_dir, "docker-compose.yml")

    # Read the docker-compose.yml file
    with open(docker_compose_loc, 'r') as stream:
        try:
            docker_compose_data = yaml.safe_load(stream)
            services = docker_compose_data.get('services', {})

            # Extract the port and URL of the service
            urls = []
            for service, config in services.items():
                ports = config.get('ports', [])
                for port in ports:
                    # Extract host port
                    host_port = port.split(':')[1]
                    urls.append(f'http://127.0.0.1:{host_port}')

            print(urls)  # Print URLs

        except yaml.YAMLError as exc:
            print(exc)
    
    return urls

def make_env(url, seed):
    ''' Function that instantiates the environment. 
    Parameters
    ----------
    url: string
        Rest API url for communication with this environment. 
    seed: integer
        Seed for random starting times of episodes in this environment.
    '''

    def _init():
        random.seed(seed)
        env = BoptestGymEnv(
            url=url,
            actions=['oveHeaPumY_u'],
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
            scenario={'electricity_price': 'dynamic'},
            predictive_period=24 * 3600,
            regressive_period=6 * 3600,
            random_start_time=True,
            excluding_periods=[(16 * 24 * 3600, 30 * 24 * 3600), (108 * 24 * 3600, 122 * 24 * 3600)],
            max_episode_length=24 * 3600,
            warmup_period=24 * 3600,
            step_period=15 * 60
        )
        env = NormalizedObservationWrapper(env)  # Add observation normalization if needed
        env = DiscretizedActionWrapper(env, n_bins_act=10)  # Add action discretization if needed

        return env

    return _init

def train_DQN_vectorized(venv, 
                         log_dir=os.path.join(utilities.get_root_path(), 'examples', 'agents', 'DQN_vectorized')):
    '''Method to train DQN agent using vectorized environment. 

    Parameters
    ----------
    venv: stable_baselines3.common.vec_env.SubprocVecEnv
        vectorized environment to be learned. 

    '''

    # Create logging directory if not exists. Monitoring data and agent model will be stored here
    os.makedirs(log_dir, exist_ok=True)

    # Modify the environment to include the callback
    venv = VecMonitor(venv=venv, filename=os.path.join(log_dir,'monitor.csv'))
            
    # Create the callback: evaluate with one episode after 100 steps for training. We keep it very short for testing.
    # When using multiple environments, each call to ``env.step()`` will effectively correspond to ``n_envs`` steps. 
    # To account for that, you can use ``eval_freq = eval_freq/venv.num_envs``
    eval_freq = 100
    eval_callback = EvalCallback(venv, best_model_save_path=log_dir, log_path=log_dir, 
                                 eval_freq=int(eval_freq/venv.num_envs), n_eval_episodes=1, deterministic=True)

    # Try to find CUDA core since it's optimized for parallel computing tasks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate an RL agent with DQN
    model = DQN('MlpPolicy', venv, verbose=1, gamma=0.99, learning_rate=5e-4,
                batch_size=24, seed=123456, buffer_size=365 * 24,
                learning_starts=24, train_freq=1, exploration_initial_eps=1.0,
                exploration_final_eps=0.01, exploration_fraction=0.1, device=device)
    
    # set up logger
    new_logger = configure(log_dir, ['csv'])
    model.set_logger(new_logger)

    # Main training loop
    model.learn(total_timesteps=100, callback=eval_callback)

if __name__ == '__main__':

    boptest_root = "./"

    # Get the argument from command line when use Linux
    if len(sys.argv) >= 2:
        boptest_root_dir = sys.argv[1]
    else:
        boptest_root_dir = boptest_root

    # Use URLs obtained from docker-compose.yml
    urls = generate_urls_from_yml(boptest_root_dir=boptest_root_dir)

    # Create BOPTEST-Gym environment replicas
    envs = [make_env(url) for url in urls]
    
    # Create a vectorized environment using SubprocVecEnv
    venv = SubprocVecEnv(envs)
    
    # Train vectorized environment
    train_DQN_vectorized(venv)






