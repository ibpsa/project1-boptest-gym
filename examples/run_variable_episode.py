'''
Module to shortly train an A2C agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script. This example is 
rather used for testing to prove the use of variable episode lengths when
training an agent. 

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

def train_A2C_with_variable_episode(start_time_tests    = [31*24*3600, 304*24*3600], 
                                    episode_length_test = 14*24*3600,
                                    log_dir = os.path.join(utilities.get_root_path(), 
                                        'examples', 'agents', 'variable_episode_A2C'),
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
    
    # Define custom child class:
    class BoptestGymEnvVariableEpisodeLength(BoptestGymEnvRewardWeightCost):
        '''Boptest gym environment that redefines the reward function to 
        weight more the operational cost when compared with the default reward
        function. 
        
        '''
        
        def compute_truncated(self, res, reward=None, 
                              objective_integrand_threshold=0.1):
            '''Custom method to determine that the episode is truncated not only 
            when the maximum episode length is exceeded but also when the 
            objective integrand overpasses a certain threshold. The latter is
            useful to early terminate agent strategies that do not work, hence
            avoiding unnecessary steps and leading to improved sampling 
            efficiency. 
            
            Returns
            -------
            truncated: boolean
                Boolean indicating whether the episode is truncated or not.  
            
            '''
            
            truncated =  (res['time'] >= self.start_time + self.max_episode_length)\
                         or \
                         (self.objective_integrand >= objective_integrand_threshold)
            
            return truncated
        
    excluding_periods = []
    for start_time_test in start_time_tests:
        excluding_periods.append((start_time_test,start_time_test+episode_length_test))
    # Summer period (from June 21st till September 22nd). 
    # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  
    
    # Use only six hours as max_episode_length to have more callbacks
    env = BoptestGymEnvVariableEpisodeLength(url                   = url,
                                             actions               = ['oveHeaPumY_u'],
                                             observations          = {'reaTZon_y':(280.,310.)}, 
                                             random_start_time     = True,
                                             excluding_periods     = excluding_periods,
                                             max_episode_length    = 6*3600, 
                                             warmup_period         = 3*3600,
                                             step_period           = 900)
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    
    # Create the callback: check every 10 steps. We keep it very short for testing 
    callback = SaveAndTestCallback(env, check_freq=10, log_dir=log_dir)
    
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
    env, model, start_time_tests = train_A2C_with_variable_episode()
    
    