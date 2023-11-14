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

seed = 123456

# Seed for random starting times of episodes
random.seed(seed)

boptest_root = "./"  # You can define boptest_root_dir here when use IDLE

# Get the argument from command line when use Linux
if len(sys.argv) >= 2:
    boptest_root_dir = sys.argv[1]
else:
    boptest_root_dir = boptest_root


def generate_urls_from_yml(boptest_root_dir=boptest_root_dir):
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

# Create a function to initialize the environment
def make_env(url):
    def _init():
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


if __name__ == '__main__':
    # Use URLs obtained from docker-compose.yml
    urls = generate_urls_from_yml(boptest_root_dir)
    if urls:  # Make sure the urls list is not empty
        envs = [make_env(url) for url in urls]

        # Create a parallel environment using SubprocVecEnv
        vec_env = SubprocVecEnv(envs)

        # Define logging directory. Monitoring data and agent model will be stored here
        log_dir = os.path.join(utilities.get_root_path(), 'examples', 'agents', 'DQN_vectorized')
        os.makedirs(log_dir, exist_ok=True)
    
        # Modify the environment to include the callback
        vec_env = VecMonitor(venv=vec_env, filename=os.path.join(log_dir,'monitor.csv'))
                
        # Create the callback: evaluate with one episode after 100 steps for training. We keep it very short for testing.
        # When using multiple environments, each call to ``env.step()`` will effectively correspond to ``n_envs`` steps. 
        # To account for that, you can use ``eval_freq = eval_freq/len(envs)``
        eval_freq = 100
        eval_callback = EvalCallback(vec_env, best_model_save_path=log_dir, log_path=log_dir, 
                                     eval_freq=int(eval_freq/len(envs)), n_eval_episodes=1, deterministic=True)

        # Try to find CUDA core since it's optimized for parallel computing tasks
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate an RL agent with DQN
        model = DQN('MlpPolicy', vec_env, verbose=1, gamma=0.99, learning_rate=5e-4,
                    batch_size=24, seed=123456, buffer_size=365 * 24,
                    learning_starts=24, train_freq=1, exploration_initial_eps=1.0,
                    exploration_final_eps=0.01, exploration_fraction=0.1, device=device)
        
        # set up logger
        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        # Main training loop
        model.learn(total_timesteps=100, callback=eval_callback)
    else:
        print("No URLs found. Please check your docker-compose.yml file.")





