'''
Module to train and test a PPO2 agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script. 

'''

from boptestGymEnv import BoptestGymEnvRewardWeightCost, NormalizedActionWrapper, \
    NormalizedObservationWrapper, SaveOnBestTrainingRewardCallback
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from examples.test_and_plot import test_agent
from testing import utilities
import random
import os

url = 'http://127.0.0.1:5000'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)

def train_PPO2(start_time_tests   = [31*24*3600, 304*24*3600], 
              episode_length_test = 14*24*3600, 
              load                = False,
              case                = 'simple',
              training_timesteps  = 1e6):
    '''Method to train (or load a pre-trained) PPO2 agent. Testing periods 
    have to be introduced already here to not use these during training. 
    
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
    load : boolean
        Boolean indicating whether the algorithm is loaded (True) or 
        needs to be trained (False)
    case : string
        Case to be tested.
    training_timesteps : integer
        Total number of timesteps used for training
     
    '''
    excluding_periods = []
    for start_time_test in start_time_tests:
        excluding_periods.append((start_time_test,start_time_test+episode_length_test))
    # Summer period (from June 21st till September 22nd). 
    # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  
    
    env = BoptestGymEnvRewardWeightCost(url                   = url,
                                        actions               = ['oveHeaPumY_u'],
                                        observations          = {'reaTZon_y':(280.,310.)}, 
                                        random_start_time     = True,
                                        excluding_periods     = excluding_periods,
                                        max_episode_length    = 1*24*3600,
                                        warmup_period         = 3*3600,
                                        step_period           = 900)
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    # Create a log directory
    log_dir = os.path.join(utilities.get_root_path(), 'examples', 
        'agents', 'PPO2_{}_{:.0e}_logdir'.format(case,training_timesteps))
    os.makedirs(log_dir, exist_ok=True)
    
    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    
    # Create the callback: check every 1000 steps 
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    
    model = PPO2('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed,
                tensorboard_log=log_dir, n_cpu_tf_sess=1)
    
    if not load: 
        model.learn(total_timesteps=int(training_timesteps), callback=callback)
        # Save the agent
        model = PPO2.save(save_path=os.path.join(log_dir,'last_model')) 
    else:
        # Load the trained agent
        model = PPO2.load(load_path=os.path.join(log_dir,'last_model'))
    
    return env, model, start_time_tests
        
def test_feb(env, model, start_time_tests, 
             episode_length_test, warmup_period_test, plot=False):
    ''' Perform test in February
    
    '''
    
    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[0], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      plot=plot)
    return observations, actions, rewards, kpis

def test_nov(env, model, start_time_tests, 
             episode_length_test, warmup_period_test, plot=False):
    ''' Perform test in November
    
    '''

    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[1], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      plot=plot)
    return observations, actions, rewards, kpis

if __name__ == "__main__":
    env, model, start_time_tests = train_PPO2(load=True)
    episode_length_test = 14*24*3600
    warmup_period_test  = 3*24*3600
    plot = True
    test_feb(env, model, start_time_tests, episode_length_test, warmup_period_test, plot)
    test_nov(env, model, start_time_tests, episode_length_test, warmup_period_test, plot)
    