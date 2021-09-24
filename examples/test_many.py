'''
Test many RL cases to generate KPIs

'''

from examples.train_RL import train_RL, test_peak, test_typi

import matplotlib.pyplot as plt
from scipy import interpolate
from gym.core import Wrapper
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import json
import os



if __name__ == "__main__":
    render = True
    plot = not render # Plot does not work together with render
    
    env, model, start_time_tests, log_dir = train_RL(algorithm='DQN', mode='train', case='D', training_timesteps=1e6, render=render, expert_traj=os.path.join('trajectories','expert_traj_disc_28.npz'))
    
    warmup_period_test  = 7*24*3600
    episode_length_test = 14*24*3600
    kpis_to_file = True

    test_peak(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, kpis_to_file, plot)
    test_typi(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, kpis_to_file, plot)
    

    
    