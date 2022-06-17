'''
Created on May 13, 2021

@author: Javier Arroyo

Generates an expert trajectory for pretraining.

'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from boptestGymEnv import DiscretizedActionWrapper
# from stable_baselines3.gail.dataset.record_expert import generate_expert_traj
from stable_baselines3 import A2C, DQN
from gym.core import Wrapper
from examples import train_RL

class ExpertModelCont(A2C):
    '''Simple proportional controller for this emulator that works as an 
    expert to pretrain the RL algorithms. Case with continuous actions. 
    The generated expert dataset works e.g. with A2C and SAC.   
    
    '''
    def __init__(self, env, TSet=22+273.15, k=1):
        super(ExpertModelCont, self).__init__(env=env,policy='MlpPolicy')

        self.env   = env
        self.TSet  = TSet
        self.k     = k 
    
    def predict(self, obs, deterministic=True):
        self.env.envs[0]
        self.env.envs[0].measurement_vars
        # Find index
        i_obs = self.env.envs[0].observations.index('reaTZon_y')
        # Rescale
        l = self.env.envs[0].lower_obs_bounds[i_obs]
        u = self.env.envs[0].upper_obs_bounds[i_obs]
        TZon = l + ((1+obs[0][i_obs])*(u-l)/2)
        # Compute control between -1 and 1 since env is normalized
        return np.array([min(1,max(-1,self.k*(self.TSet-TZon)))]), None
    
    def get_env(self):
        return self.env.envs[0]

class ExpertModelDisc(DQN):
    '''Simple proportional controller for this emulator that works as an 
    expert to pretrain the RL algorithms. The generated expert dataset 
    works e,g, with DQN. 
    
    '''
    def __init__(self, env, 
                 n_bins_act = 10, 
                 TSet=22+273.15, k=1):
        self.env        = DiscretizedActionWrapper(env,n_bins_act=n_bins_act)
        self.n_bins_act = n_bins_act
        self.TSet       = TSet
        self.k          = k 
        self.act_vals   = np.arange(n_bins_act+1)
    
    def predict(self, obs, deterministic=True):
        self.env
        self.env.measurement_vars
        # Find index
        i_obs = self.env.observations.index('reaTZon_y')
        # Rescale
        l = self.env.lower_obs_bounds[i_obs]
        u = self.env.upper_obs_bounds[i_obs]
        TZon = l + ((1+obs[i_obs])*(u-l)/2)
        # Compute control 
        value = self.k*(self.TSet-TZon)
        # Transform from [-1,1] to [0,10] since env is discretized
        value = 5*value + 5
        # Bound result
        value = min(10,max(0,value))
        return self.find_nearest_action(value), _
    
    def get_env(self):
        return self.env
    
    def find_nearest_action(self, value):
        idx = (np.abs(self.act_vals - value)).argmin()
        return self.act_vals[idx]


if __name__ == "__main__":
    n_days = 0.1
    cont_disc = 'cont'
    env, _, start_time_tests, log_dir = train_RL.train_RL(max_episode_length = n_days*24*3600, 
                                                          mode='empty', 
                                                          case='D', 
                                                          render=True)
    
    # Set expert trajectory to start the first day of February    
    start_year      = '2021-01-01 00:00:00'
    start           = '2021-02-01 00:00:00'
    start_time      = (pd.Timestamp(start)-pd.Timestamp(start_year)).total_seconds()    
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time   = False
        env.unwrapped.start_time          = start_time
    else:
        env.random_start_time   = False
        env.start_time          = start_time
    
    # Instantiate expert model. Distinguish between continuous or discrete
    if cont_disc == 'cont':
        expert_model = ExpertModelCont(env)
    elif cont_disc == 'disc':
        expert_model = ExpertModelDisc(env)
    
    # Generate data and save in a numpy archive named `expert_traj.npz`
    print('Generating expert data...')
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv

    model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=123456,
                learning_rate=7e-4, n_steps=4, ent_coef=1,
                tensorboard_log=log_dir)

    rollouts = rollout.generate_trajectories(
        policy=expert_model,
        venv=DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        sample_until=rollout.make_sample_until(None, 1),
    )
    transitions = rollout.flatten_trajectories(rollouts)

    print(
        f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
    After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
    The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
    """
    )

    print('implementing behavior cloning...')
    from imitation.algorithms import bc
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        expert_data=transitions,
    )

    from stable_baselines3.common.evaluation import evaluate_policy

    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=1)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward after training: {reward_after_training}")

    traj_name = os.path.join('trajectories',
                             'expert_traj_{}_{}'.format(cont_disc, n_days))
    print(traj_name)
    # generate_expert_traj(expert_model,
    #                      traj_name,
    #                      n_episodes=1)
    #

    # plt.savefig(traj_name+'.pdf', bbox_inches='tight')
    