'''
Module to shortly train an A2C agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script. This example is 
rather used for testing to prove the use of a callback that monitors model
performance and saves a model upon improved performance. 

'''

from boptestGymEnv import BoptestGymEnvRewardWeightCost, NormalizedActionWrapper, NormalizedObservationWrapper, SaveAndTestCallback
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from testing import utilities
import random
import os

url = 'http://127.0.0.1'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)

def train_A2C_with_callback(start_time_tests    = [31*24*3600, 304*24*3600], 
                            episode_length_test = 14*24*3600,
                            log_dir = os.path.join(utilities.get_root_path(), 
                                'examples', 'agents', 'monitored_A2C'),
                            tensorboard_log     = os.path.join('results')):
    '''Method to train an A2C agent using a callback to save the model 
    upon performance improvement.  
    
    Parameters
    ----------
    start_time_tests : list of integers
        Time in seconds from the beginning of the year that will be used 
        for testing. These periods should be excluded in the training 
        process. By default the first day of February and the first day of
        November are used. 
    episode_length_test : integer
        Number of seconds indicating the length of the testing periods. By
        default two weeks are reserved for testing.  
    log_dir : string
        Directory where monitoring data and best trained model are stored.
    tensorboard_log : path
        Path to directory to load tensorboard logs.
    
    '''
    
    excluding_periods = []
    for start_time_test in start_time_tests:
        excluding_periods.append((start_time_test,start_time_test+episode_length_test))
    # Summer period (from June 21st till September 22nd). 
    # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  
    
    # Use only one hour episode to have more callbacks
    env = BoptestGymEnvRewardWeightCost(url                   = url,
                                        actions               = ['oveHeaPumY_u'],
                                        observations          = {'reaTZon_y':(280.,310.)}, 
                                        random_start_time     = True,
                                        excluding_periods     = excluding_periods,
                                        max_episode_length    = 1*3600, 
                                        warmup_period         = 3*3600,
                                        step_period           = 900)
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    
    # Create the callback: check every 10 steps. We keep it very short for testing 
    callback = SaveAndTestCallback(env=env, check_freq=10, log_dir=log_dir)
    
    # Initialize the agent
    model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed,
                tensorboard_log=tensorboard_log)
    
    # set up logger
    new_logger = configure(log_dir, ['csv'])
    model.set_logger(new_logger)

    # Train the agent with callback for saving
    model.learn(total_timesteps=int(100), callback=callback)
    
    return env, model, start_time_tests

if __name__ == "__main__":
    env, model, start_time_tests = train_A2C_with_callback()
    
    