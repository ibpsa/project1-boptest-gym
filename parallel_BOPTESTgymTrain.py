import yaml
import torch

from stable_baselines3 import DQN, PPO, DDPG, SAC, TD3, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper


# 读取docker-compose.yml文件
with open("docker-compose.yml", 'r') as stream:
    try:
        docker_compose_data = yaml.safe_load(stream)
        services = docker_compose_data.get('services', {})

        # 提取服务的端口和URL
        urls = []
        for service, config in services.items():
            ports = config.get('ports', [])
            for port in ports:
                # 提取主机端口
                host_port = port.split(':')[1]
                urls.append(f'http://127.0.0.1:{host_port}')

        print(urls)  # 打印服务URLs

    except yaml.YAMLError as exc:
        print(exc)


# 创建一个函数来初始化环境
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
        env = NormalizedObservationWrapper(env)  # 如果需要的话添加观测归一化
        env = DiscretizedActionWrapper(env, n_bins_act=10)  # 如果需要的话添加动作离散化

        return env

    return _init


if __name__ == '__main__':
    # 使用从docker-compose.yml文件中获取的URLs
    if urls:  # 确保urls列表不是空的
        envs = [make_env(url) for url in urls]

        # 使用 SubprocVecEnv 创建并行环境
        vec_env = SubprocVecEnv(envs)

        # 创建 DQN 模型
        log_dir = "./vec_dqn_log/"
        eval_callback = EvalCallback(vec_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=5000)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate an RL agent with DQN
        model = DQN('MlpPolicy', vec_env, verbose=1, gamma=0.99, learning_rate=5e-4,
                    batch_size=24, seed=123456, buffer_size=365 * 24,
                    learning_starts=24, train_freq=1, exploration_initial_eps=1.0,
                    exploration_final_eps=0.01, exploration_fraction=0.1, tensorboard_log=log_dir, device=device)
        # Main training loop
        model.learn(total_timesteps=500000, callback=eval_callback)
    else:
        print("No URLs found. Please check your docker-compose.yml file.")





