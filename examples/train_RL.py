'''
Module to train and test a RL agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script.  
To load the ExpertDataset it's needed to comment the first line in stable_baselines\gail\_init_.py
from stable_baselines3.gail.model import GAIL
If further issues are encountered related to the np.ndarrais for pretraining, it may happen that
numpy is installed twice. Check:
https://stackoverflow.com/questions/54943168/problem-with-tensorflow-tf-sessionrun-wrapper-expected-all-values-in-input-dic

'''

from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, \
    NormalizedObservationWrapper, SaveAndTestCallback, DiscretizedActionWrapper
# from stable_baselines3.gail import ExpertDataset
from stable_baselines3 import A2C, SAC, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from examples.test_and_plot import test_agent
from collections import OrderedDict
from testing import utilities
import requests
import random
import os

url = 'http://127.0.0.1'
seed = 123456

# Seed for random starting times of episodes
random.seed(seed)

def train_RL(algorithm           = 'SAC',
             start_time_tests    = [(23-7)*24*3600, (115-7)*24*3600], 
             episode_length_test = 14*24*3600, 
             warmup_period       = 1*24*3600,
             max_episode_length  = 7*24*3600,
             mode                = 'train',
             case                = 'simple',
             training_timesteps  = 3e5,
             render              = False,
             expert_traj         = None, 
             model_name          = 'last_model'):
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
    mode : string
        Either train, load, or empty.
    case : string
        Case to be tested.
    training_timesteps : integer
        Total number of timesteps used for training
    render : boolean
        If true, it renders every episode while training.
    expert_traj : string
        Path to expert trajectory in .npz format. If not None, the agent 
        will be pretrained using behavior cloning with these data. 
    model_name : string
        Name of the model to be saved or loaded. 
            
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
        def get_reward(self):
            '''Custom reward function
            
            '''
            
            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
            
            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot']*12.*16. + 100*kpis['tdis_tot']
            
            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)
            
            self.objective_integrand = objective_integrand
            
            return reward
    
    if case == 'simple':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            testcase              = 'bestest_hydronic_heat_pump',
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
                            testcase              ='bestest_hydronic_heat_pump',
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
                            testcase              ='bestest_hydronic_heat_pump',
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
                            testcase              ='bestest_hydronic_heat_pump',
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
                            step_period           = 1800,
                            render_episodes       = render,
                            log_dir               = log_dir)
        
    if case == 'D':
        env = BoptestGymEnvCustomReward(
                            url                   = url,
                            testcase              ='bestest_hydronic_heat_pump',
                            actions               = ['oveHeaPumY_u'],
                            observations          = OrderedDict([('time',(0,604800)),
                                                     ('reaTZon_y',(280.,310.)),
                                                     ('TDryBul',(265,303)),
                                                     ('HDirNor',(0,862)),
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
                            step_period           = 900,
                            render_episodes       = render,
                            log_dir               = log_dir)
    
    env = NormalizedObservationWrapper(env)
    env = NormalizedActionWrapper(env)  
    
    # Modify the environment to include the callback
    env = Monitor(env=env, filename=os.path.join(log_dir,'monitor.csv'))
    
    if mode == 'train': 
        
        # Define RL agent
        if 'SAC' in algorithm:
            model = SAC('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=3e-4, batch_size=96, ent_coef='auto',
                        buffer_size=365*96, learning_starts=96, train_freq=1,
                        tensorboard_log=log_dir)
    
        elif 'A2C' in algorithm:
            model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=1e-6, n_steps=4, ent_coef=0,
                        tensorboard_log=log_dir)
            
        elif 'DQN' in algorithm:
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=5e-4, batch_size=24, 
                        buffer_size=365*24, learning_starts=24, train_freq=1,
                        tensorboard_log=log_dir)
        
        if expert_traj is not None:
            # Do not shuffle (randomize) to obtain deterministic result
            dataset = ExpertDataset(expert_path=expert_traj, randomize=False,
                                    traj_limitation=1, batch_size=96)
            model.pretrain(dataset, n_epochs=1000)
        
        # Create the callback test and save the agent while training
        callback = SaveAndTestCallback(env, check_freq=1e10, save_freq=1e4,
                                       log_dir=log_dir, test=False)
        
        # set up logger
        new_logger = configure(log_dir, ['csv'])
        model.set_logger(new_logger)

        # Main training loop
        model.learn(total_timesteps=int(training_timesteps), callback=callback)
        # Save the agent
        model.save(os.path.join(log_dir,model_name))
        
    elif mode == 'load':
        # Load the trained agent
        if 'SAC' in algorithm:
            model = SAC.load(os.path.join(log_dir,model_name))
        elif 'A2C' in algorithm:
            model = A2C.load(os.path.join(log_dir,model_name))
        elif 'DQN' in algorithm:
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN.load(os.path.join(log_dir,model_name))
            
    elif mode == 'empty':
        model = None
    
    else:
        raise ValueError('mode should be either train, load, or empty')
    
    return env, model, start_time_tests, log_dir
        
def test_peak(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), model_name='last_model', 
              save_to_file=False, plot=False):
    ''' Perform test in peak heat period (February). 
    
    '''

    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[0], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir, model_name=model_name,
                                                      save_to_file=save_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis

def test_typi(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), model_name='last_model', 
              save_to_file=False, plot=False):
    ''' Perform test in typical heat period (November)
    
    '''

    observations, actions, rewards, kpis = test_agent(env, model, 
                                                      start_time=start_time_tests[1], 
                                                      episode_length=episode_length_test,
                                                      warmup_period=warmup_period_test,
                                                      log_dir=log_dir, model_name=model_name,
                                                      save_to_file=save_to_file,
                                                      plot=plot)
    return observations, actions, rewards, kpis

if __name__ == "__main__":
    render = True
    plot = not render # Plot does not work together with render
    
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='A', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='B', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='C', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='DQN', mode='load', case='D', training_timesteps=1e6, render=render)
    
    env, model, start_time_tests, log_dir = train_RL(algorithm='A2C', mode='train', case='D', training_timesteps=1e6, render=render, expert_traj=os.path.join('trajectories','expert_traj_cont_28.npz'))
    
    warmup_period_test  = 7*24*3600
    episode_length_test = 14*24*3600
    save_to_file = True

    test_peak(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, save_to_file, plot)
    test_typi(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, save_to_file, plot)
    
