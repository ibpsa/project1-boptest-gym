'''
Module to train and test an A2C agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script.  

'''

from boptestGymEnv import BoptestGymEnvRewardWeightCost, NormalizedActionWrapper, \
    NormalizedObservationWrapper, SaveOnBestTrainingRewardCallback
from stable_baselines import A2C
from stable_baselines.bench import Monitor
from examples.test_and_plot import test_agent
from collections import OrderedDict
from testing import utilities
import random
import os

url = 'http://127.0.0.1:5000'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)

def train_A2C(start_time_tests    = [31*24*3600, 304*24*3600], 
              episode_length_test = 14*24*3600, 
              load                = False,
              case                = 'simple',
              training_timesteps  = 1e6):
    '''Method to train (or load a pre-trained) A2C agent. Testing periods 
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
    
    if case == 'simple':
        env = BoptestGymEnvRewardWeightCost(url                   = url,
                                            actions               = ['oveHeaPumY_u'],
                                            observations          = OrderedDict([('reaTZon_y',(280.,310.))]), 
                                            random_start_time     = True,
                                            excluding_periods     = excluding_periods,
                                            max_episode_length    = 1*24*3600,
                                            warmup_period         = 3*3600,
                                            step_period           = 900)
    elif case == 'A':
        env = BoptestGymEnvRewardWeightCost(url                   = url,
                                            actions               = ['oveHeaPumY_u'],
                                            observations          = OrderedDict([('time',(0,604800)),
                                                                     ('reaTZon_y',(280.,310.)),
                                                                     ('PriceElectricPowerHighlyDynamic',(-0.5,0.14))]), 
                                            scenario              = {'electricity_price':'highly_dynamic'},
                                            forecasting_period    = 0, 
                                            random_start_time     = True,
                                            excluding_periods     = excluding_periods,
                                            max_episode_length    = 1*24*3600,
                                            warmup_period         = 1*24*3600,
                                            step_period           = 900)
    if case == 'B':
        env = BoptestGymEnvRewardWeightCost(url                   = url,
                                            actions               = ['oveHeaPumY_u'],
                                            observations          = OrderedDict([('time',(0,604800)),
                                                                     ('reaTZon_y',(280.,310.)),
                                                                     ('PriceElectricPowerHighlyDynamic',(-0.5,0.14)),
                                                                     ('LowerSetp[1]',(280.,310.)),
                                                                     ('UpperSetp[1]',(280.,310.))]), 
                                            forecasting_period    = 0, 
                                            scenario              = {'electricity_price':'highly_dynamic'},
                                            random_start_time     = True,
                                            excluding_periods     = excluding_periods,
                                            max_episode_length    = 1*24*3600,
                                            warmup_period         = 1*24*3600,
                                            step_period           = 900)
    if case == 'C':
        env = BoptestGymEnvRewardWeightCost(url                   = url,
                                            actions               = ['oveHeaPumY_u'],
                                            observations          = OrderedDict([('time',(0,604800)),
                                                                     ('reaTZon_y',(280.,310.)),
                                                                     ('PriceElectricPowerHighlyDynamic',(-0.5,0.14)),
                                                                     ('LowerSetp[1]',(280.,310.)),
                                                                     ('UpperSetp[1]',(280.,310.))]), 
                                            forecasting_period    = 3*3600, 
                                            scenario              = {'electricity_price':'highly_dynamic'},
                                            random_start_time     = True,
                                            excluding_periods     = excluding_periods,
                                            max_episode_length    = 1*24*3600,
                                            warmup_period         = 1*24*3600,
                                            step_period           = 900)    
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    # Create a log directory
    log_dir = os.path.join(utilities.get_root_path(), 'examples', 
        'agents', 'A2C_{}_{:.0e}_logdir'.format(case,training_timesteps))
    os.makedirs(log_dir, exist_ok=True)
    
    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    
    # Create the callback: check every 1000 steps 
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    
    model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed,
                tensorboard_log=log_dir, n_cpu_tf_sess=1)
    
    if not load: 
        model.learn(total_timesteps=int(training_timesteps), callback=callback)
        # Save the agent
        model = A2C.save(save_path=os.path.join(log_dir,'last_model'))
    else:
        # Load the trained agent
        model = A2C.load(load_path=os.path.join(log_dir,'last_model'))
    
    return env, model, start_time_tests
        
def test_feb(env, model, start_time_tests, episode_length_test, 
             warmup_period_test, kpis_to_file=False, plot=False):
    ''' Perform test in February
    
    '''

    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[0], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      kpis_to_file=kpis_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis

def test_nov(env, model, start_time_tests, episode_length_test, 
             warmup_period_test, kpis_to_file=False, plot=False):
    ''' Perform test in November
    
    '''

    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[1], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      kpis_to_file=kpis_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis

if __name__ == "__main__":
    env, model, start_time_tests = train_A2C(load=True, case='B', training_timesteps=1e6)
    episode_length_test = 14*24*3600
    warmup_period_test  = 1*24*3600
    kpis_to_file = True
    plot = True
    test_feb(env, model, start_time_tests, episode_length_test, warmup_period_test, kpis_to_file, plot)
    test_nov(env, model, start_time_tests, episode_length_test, warmup_period_test, kpis_to_file, plot)
    