'''
Module to train and test an A2C agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script.  

'''

from boptestGymEnv import BoptestGymEnvRewardWeightCost, NormalizedActionWrapper, NormalizedObservationWrapper, SaveOnBestTrainingRewardCallback
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv
from examples.test_and_plot import test_agent
from stable_baselines.bench import Monitor
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
              log_dir = os.path.join(utilities.get_root_path(), 
                                     'examples', 'agents', 'A2C_dynprice3')):
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
     
    '''
    excluding_periods = []
    for start_time_test in start_time_tests:
        excluding_periods.append((start_time_test,start_time_test+episode_length_test))
    # Summer period (from June 21st till September 22nd). 
    # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  
    
    env = BoptestGymEnvRewardWeightCost(url                   = url,
                                        actions               = ['oveHeaPumY_u'],
                                        observations          = {'reaTZon_y':   (280.,310.),
                                                                 'LowerSetp[1]':(280.,310.),
                                                                 'UpperSetp[1]':(280.,310.),
                                                                 #'TDryBul':     (250.,310.),
                                                                 #'HGloHor':     (0.,  1000.),
                                                                 'PriceElectricPowerHighlyDynamic': (-0.5,0.14)}, 
                                        random_start_time     = True,
                                        excluding_periods     = excluding_periods,
                                        forecasting_period    = 0*3600,
                                        max_episode_length    = 24*3600,
                                        scenario              = {'electricity_price':'highly_dynamic'},
                                        warmup_period         = 3*3600,
                                        Ts                    = 900)
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    # Modify the environment to include the callback when learning
    if not load:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    else:
        # This is to not overwrite the monitor.csv file
        os.makedirs(log_dir+'_testing', exist_ok=True)
        env = Monitor(env=env, filename=os.path.join(log_dir+'_testing','monitor.csv'))
    
    # Create the callback: check every 10 steps. We keep it very short for testing 
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    
    # Initialize the agent
    model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed,
                tensorboard_log=os.path.join('results'))
    
    if os.path.exists(os.path.join(log_dir,'best_model')):
        # Try to load a pre-trained agent. The A2C model requires a VecEnv.
        # Wrap it in a DummyVecEnv to avoid error
        # If no exists just go with the newly created agent
        env = DummyVecEnv([lambda: env])
        model.n_envs = 1
        model = A2C.load(os.path.join(log_dir,'best_model'), env=env, 
                         verbose=1, gamma=0.99, seed=seed,
                         tensorboard_log=os.path.join('results'))

    if not load: 
        model.learn(total_timesteps=int(5e4), callback=callback)
        # Save the agent
        model.save(save_path=os.path.join(utilities.get_root_path(),'examples',
                                      'agents','a2c_dynprice'))
    else:
        # Load the trained agent
        model = A2C.load(os.path.join(log_dir,'best_model'))
    
    return env, model, start_time_tests
        
def test_jan(env, model, start_time_tests, 
             episode_length_test, warmup_period_test, plot=True):
    ''' Perform test in January
    
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
    env, model, start_time_tests = train_A2C(load=True)
    episode_length_test = 5*24*3600
    warmup_period_test  = 3*3600
    plot = True
    test_jan(env, model, start_time_tests, episode_length_test, warmup_period_test, plot)
    test_nov(env, model, start_time_tests, episode_length_test, warmup_period_test, plot)
    