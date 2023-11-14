import os
import sys
import yaml
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper


boptest_root = "./"  # You can define boptest_root_dir here when use IDLE

# Get the argument from command line when use Linux
if len(sys.argv) >= 2:
    boptest_root_dir = sys.argv[1]
else:
    boptest_root_dir = boptest_root

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
            max_episode_length=14 * 24 * 3600,
            warmup_period=24 * 3600,
            step_period=15 * 60
        )
        env = NormalizedObservationWrapper(env)  # Add observation normalization if needed
        env = DiscretizedActionWrapper(env, n_bins_act=10)  # Add action discretization if needed

        return env

    return _init


if __name__ == '__main__':
    # Use URLs obtained from docker-compose.yml
    if urls:  # Make sure the urls list is not empty
        envs = [make_env(url) for url in urls]

        # Create a parallel environment using SubprocVecEnv
        vec_env = SubprocVecEnv(envs)

        # Example: Create a DQN model
        log_dir = "./vec_dqn_log/"
        eval_callback = EvalCallback(vec_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=5000)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate an RL agent with DQN
        model = DQN('MlpPolicy', vec_env, verbose=1, gamma=0.99, learning_rate=5e-4,
                    batch_size=24, seed=123456, buffer_size=365 * 24,
                    learning_starts=24, train_freq=1, exploration_initial_eps=1.0,
                    exploration_final_eps=0.01, exploration_fraction=0.1, device=device)
        # Main training loop
        model.learn(total_timesteps=500000, callback=eval_callback)
    else:
        print("No URLs found. Please check your docker-compose.yml file.")





