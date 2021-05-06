'''
Module to train and test a RL agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script.  

'''

from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, \
    NormalizedObservationWrapper, SaveAndTestCallback, DiscretizedActionWrapper
from stable_baselines import A2C, SAC, DQN
from stable_baselines.bench import Monitor
from examples.test_and_plot import test_agent
from collections import OrderedDict
from testing import utilities
import requests
import random
import os

url = 'http://127.0.0.1:5000'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)

def train_RL(algorithm           = 'SAC',
             start_time_tests    = [(45-7)*24*3600, (310-7)*24*3600], 
             episode_length_test = 14*24*3600, 
             warmup_period       = 1*24*3600,
             max_episode_length  = 7*24*3600,
             load                = False,
             case                = 'simple',
             training_timesteps  = 3e5,
             render              = False):
    '''Method to train (or load a pre-trained) A2C agent. Testing periods 
    have to be introduced already here to not use these during training. 
    
    Parameters
    ----------
    start_time_tests : list of integers
        Time in seconds from the beginning of the year that will be used 
        for testing. These periods should be excluded in the training 
        process. By default the peak and typical heat periods for the 
        bestest hydronic case with a heat pump are used. 
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
    render : boolean
        If true, it renders every episode while training.
        
    '''
    
    excluding_periods = []
    for start_time_test in start_time_tests:
        excluding_periods.append((start_time_test,
                                  start_time_test+episode_length_test))
    # Summer period (from June 21st till September 22nd). 
    # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  
    
    # Create a log directory
    log_dir = os.path.join(utilities.get_root_path(), 'examples', 
        'agents', '{}_{}_{:.0e}_logdir'.format(algorithm,case,training_timesteps))
    log_dir = log_dir.replace('+', '')
    os.makedirs(log_dir, exist_ok=True)
    
    # Redefine reward function
    class BoptestGymEnvCustomReward(BoptestGymEnv):
        '''Define a custom reward for this building
        
        '''
        def compute_reward(self):
            '''Custom reward function
            
            '''
            
            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi'.format(self.url)).json()
            
            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot'] + 10*kpis['tdis_tot']
            
            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)
            
            self.objective_integrand = objective_integrand
            
            return reward
    
    if case == 'simple':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            actions               = ['oveHeaPumY_u'],
                            observations          = OrderedDict([('reaTZon_y',(280.,310.))]), 
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    elif case == 'A':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            actions               = ['oveHeaPumY_u'],
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4))]), 
                            scenario              = {'electricity_price':'highly_dynamic'},
                            predictive_period     = 0, 
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    if case == 'B':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            actions               = ['oveHeaPumY_u'],
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4)),
                                                     ('LowerSetp[1]',(280.,310.)),
                                                     ('UpperSetp[1]',(280.,310.))]), 
                            predictive_period     = 0, 
                            scenario              = {'electricity_price':'highly_dynamic'},
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    if case == 'C':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            actions               = ['oveHeaPumY_u'],
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4)),
                                                     ('LowerSetp[1]',(280.,310.)),
                                                     ('UpperSetp[1]',(280.,310.))]), 
                            predictive_period     = 3*3600, 
                            scenario              = {'electricity_price':'highly_dynamic'},
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
        
    if case == 'D':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            actions               = ['oveHeaPumY_u'],
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('TDryBul',(265,303)),
                                                     ('HGloHor',(0,991)),
                                                     ('InternalGainsRad[1]',(0,219)),
                                                     ('PriceElectricPowerHighlyDynamic',(-0.4,0.4)),
                                                     ('LowerSetp[1]',(280.,310.)),
                                                     ('UpperSetp[1]',(280.,310.))]), 
                            predictive_period     = 24*3600, 
                            regressive_period     = 6*3600, 
                            scenario              = {'electricity_price':'highly_dynamic'},
                            random_start_time     = True,
                            excluding_periods     = excluding_periods,
                            max_episode_length    = max_episode_length,
                            warmup_period         = warmup_period,
                            step_period           = 3600,
                            render_episodes       = render,
                            log_dir               = log_dir)
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    
    if not load: 
        
        # Define RL agent
        if algorithm == 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=3e-4, batch_size=96, ent_coef='auto',
                        buffer_size=365*96, learning_starts=96, train_freq=1,
                        tensorboard_log=log_dir, n_cpu_tf_sess=1)
    
        elif algorithm == 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=7e-4, n_steps=4, ent_coef=1,
                        tensorboard_log=log_dir, n_cpu_tf_sess=1)
            
        elif algorithm == 'DQN':
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=5e-4, batch_size=24, 
                        buffer_size=365*24, learning_starts=24, train_freq=1,
                        tensorboard_log=log_dir, n_cpu_tf_sess=1)
        
        # Create the callback test and save the agent while training
        callback = SaveAndTestCallback(env, check_freq=10000, save_freq=10000,
                                       log_dir=log_dir, test=True)
        # Main training loop
        model.learn(total_timesteps=int(training_timesteps), callback=callback)
        # Save the agent
        model.save(os.path.join(log_dir,'last_model'))
        
    else:
        # Load the trained agent
        if algorithm == 'SAC':
            model = SAC.load(os.path.join(log_dir,'last_model'))
        elif algorithm == 'A2C':
            model = A2C.load(os.path.join(log_dir,'last_model'))
        elif algorithm == 'DQN':
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN.load(os.path.join(log_dir,'last_model'))
    
    return env, model, start_time_tests, log_dir
        
def test_peak(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), kpis_to_file=False, 
              plot=False):
    ''' Perform test in peak heat period (February). 
    
    '''

    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[0], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir,
                                                      kpis_to_file=kpis_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis

def test_typi(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), kpis_to_file=False, 
              plot=False):
    ''' Perform test in typical heat period (November)
    
    '''

    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[1], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir,
                                                      kpis_to_file=kpis_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis

if __name__ == "__main__":
    render = True
    plot = not render # Plot does not work together with render
    
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', load=True, case='A', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', load=True, case='B', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', load=True, case='C', training_timesteps=3e5, render=render)
    env, model, start_time_tests, log_dir = train_RL(algorithm='DQN', load=False, case='D', training_timesteps=1e6, render=render)
    
    warmup_period_test  = 7*24*3600
    episode_length_test = 14*24*3600
    kpis_to_file = True

    test_peak(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, kpis_to_file, plot)
    test_typi(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, kpis_to_file, plot)
    
