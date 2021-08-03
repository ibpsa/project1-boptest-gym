'''
Module to train and test a RL agent for the bestest_hydronic_heatpump 
case. This case needs to be deployed to run this script.  
To load the ExpertDataset it's needed to comment the first line in stable_baselines\gail\_init_.py
from stable_baselines.gail.model import GAIL
If further issues are encountered related to the np.ndarrais for pretraining, it may happen that
numpy is installed twice. Check:
https://stackoverflow.com/questions/54943168/problem-with-tensorflow-tf-sessionrun-wrapper-expected-all-values-in-input-dic

'''

from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper, \
    NormalizedObservationWrapper, SaveAndTestCallback, DiscretizedActionWrapper
from stable_baselines.gail import ExpertDataset
from stable_baselines import A2C, SAC, DQN
from stable_baselines.bench import Monitor
from examples.test_and_plot_RLMPC import test_agent
from collections import OrderedDict
from testing import utilities
import requests
import random
import os

url     = 'http://127.0.0.1:5000'
url_RC  = 'http://127.0.0.1:8080'
seed = 123456
# from pyfmi import load_fmu
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
             exploration_initial_eps = 0.1,
             from_model          = None,
             render              = False,
             expert_traj         = None,
             return_RC           = False):
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
        Either train, load, continue, or empty.
    case : string
        Case to be tested.
    training_timesteps : integer
        Total number of timesteps used for training
    exploration_initial_eps : integer
        Initial exploration rate
    from_model : string
        Model from which to continue learning. To be combined with 
        mode=continue. So far only supported with DQN. 
    render : boolean
        If true, it renders every episode while training.
    expert_traj : string
        Path to expert trajectory in .npz format. If not None, the agent 
        will be pretrained using behavior cloning with these data. 
    return_RC : boolean
        True to return the same environment but pointing to the RC port
        
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
    
    if return_RC:
        # Create a log directory
        log_dir_RC = os.path.join(utilities.get_root_path(), 'examples', 
            'agents', '{}_{}_{:.0e}_logdir_RC'.format(algorithm,case,training_timesteps))
        log_dir_RC = log_dir.replace('+', '')
        os.makedirs(log_dir_RC, exist_ok=True)
    else:
        env_RC = None
    
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
            objective_integrand = kpis['cost_tot']*12.*16. + 1e6*kpis['tdis_tot']
            
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
        
    if case == 'D' and return_RC:
        env_RC = BoptestGymEnvCustomReward(
                            url                   = url_RC,
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
                            log_dir               = log_dir_RC)
    
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
                        tensorboard_log=log_dir, n_cpu_tf_sess=1)
    
        elif 'A2C' in algorithm:
            model = A2C('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        learning_rate=7e-4, n_steps=4, ent_coef=1,
                        tensorboard_log=log_dir, n_cpu_tf_sess=1)
            
        elif 'DQN' in algorithm:
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed, 
                        exploration_initial_eps=exploration_initial_eps, exploration_final_eps=0.01,
                        learning_rate=5e-4, batch_size=7*96, target_network_update_freq=1,
                        buffer_size=365*96, learning_starts=96, train_freq=1,
                        tensorboard_log=log_dir, n_cpu_tf_sess=1)
        
        if expert_traj is not None:
            # Do not shuffle (randomize) to obtain deterministic result
            dataset = ExpertDataset(expert_path=expert_traj, randomize=False,
                                    traj_limitation=1, batch_size=96)
            model.pretrain(dataset, n_epochs=1000)
        
        # Create the callback test and save the agent while training
        callback = SaveAndTestCallback(env, check_freq=2e6, save_freq=10000,
                                       log_dir=log_dir, test=False)
        # Main training loop
        model.learn(total_timesteps=int(training_timesteps), callback=callback)
        # Save the agent
        model.save(os.path.join(log_dir,'last_model'))
        
    elif mode == 'load':
        # Load the trained agent
        if 'SAC' in algorithm:
            model = SAC.load(os.path.join(log_dir,'last_model'))
        elif 'A2C' in algorithm:
            model = A2C.load(os.path.join(log_dir,'last_model'))
        elif 'DQN' in algorithm:
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN.load(os.path.join(log_dir,'last_model'))
            
    elif mode == 'continue':
        if 'DQN' in algorithm:
            env = DiscretizedActionWrapper(env,n_bins_act=10)
            model = DQN.load(os.path.join(log_dir,from_model))
            model.set_env(env)
            
            # Derive the initial_step from which this model is going to learn
            initial_step = int(from_model.split('_')[1])
            
            # Deduct the steps that we've already learned
            training_timesteps_reduced = training_timesteps - initial_step
            
            # Set initial exploration rate
            model.exploration_initial_eps = model.exploration_final_eps + \
                (exploration_initial_eps - model.exploration_final_eps)*\
                (training_timesteps_reduced)/training_timesteps
            
            # Set other training settings:
            model.verbose                       = 1
            model.gamma                         = 0.99
            model.seed                          = seed 
            model.exploration_final_eps         = 0.01
            model.learning_rate                 = 5e-4
            model.batch_size                    = 7*96
            model.target_network_update_freq    = 1
            model.buffer_size                   = 365*96
            model.learning_starts               = 96
            model.train_freq                    = 1
            model.tensorboard_log               = log_dir
            model.n_cpu_tf_sess                 = 1
            
            # Create the callback test and save the agent while training
            callback = SaveAndTestCallback(env, check_freq=2e6, save_freq=10000,
                                           log_dir=log_dir, test=False, initial_step=initial_step)
            
            # Main training loop
            model.learn(total_timesteps=int(training_timesteps_reduced), callback=callback)
            # Save the agent
            model.save(os.path.join(log_dir,'last_model'))
            
    elif mode == 'empty':
        model = None
    
    else:
        raise ValueError('mode should be either train, load, continue, or empty')
    
    if return_RC:
        env_RC = NormalizedObservationWrapper(env_RC)
        env_RC = NormalizedActionWrapper(env_RC)
        env_RC = Monitor(env=env_RC, filename=os.path.join(log_dir,'monitor_RC.csv'))
        env_RC = DiscretizedActionWrapper(env_RC,n_bins_act=10)
    
    return env, model, start_time_tests, log_dir, env_RC
        
def test_peak(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), save_to_file=False, 
              plot=False, env_RC=None):
    ''' Perform test in peak heat period (January). 
    
    '''

    actions, rewards, kpis = test_agent(env, model, 
                                        start_time=start_time_tests[0], 
                                        episode_length=episode_length_test,
                                        warmup_period=warmup_period_test,
                                        log_dir=log_dir,
                                        save_to_file=save_to_file,
                                        plot=plot,
                                        env_RC=env_RC)
    return actions, rewards, kpis

def test_typi(env, model, start_time_tests, episode_length_test, 
              warmup_period_test, log_dir=os.getcwd(), save_to_file=False, 
              plot=False, env_RC=None):
    ''' Perform test in typical heat period (October?)
    
    '''

    actions, rewards, kpis = test_agent(env, model, 
                                        start_time=start_time_tests[1], 
                                        episode_length=episode_length_test,
                                        warmup_period=warmup_period_test,
                                        log_dir=log_dir,
                                        save_to_file=save_to_file,
                                        plot=plot, 
                                        env_RC=env_RC)
    return actions, rewards, kpis

if __name__ == "__main__":
    render = False
    plot = not render # Plot does not work together with render
    
    #=================================================================
    # TODO:
    # -. Print actions taken and absolute temperature
    # -. Move functionality of retrieving measurements, cInp_stp, and dist_stp into observer
    # -. Store results from both, the observer and the actual environment
    # 4. Clean up imagine API so that it does not print everything. Check first that what is printed corresponds to the states of the observer. 
    # -. Analyze rewards and returns. Why positive rewards? why nan returns?
    # 6. Include internal gains into dist_step. Is it worth it?
    # 7. Run for the two weeks and in different scenarios.  
    # 8. Create a separate script for test_and_plot to recover past functionality when testing classical RL
    # -. Check that I get the same result even when I "spoil" the create_input_object method of the state observer
    #=================================================================
    
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='A', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='B', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='SAC', mode='load', case='C', training_timesteps=3e5, render=render)
    #env, model, start_time_tests, log_dir = train_RL(algorithm='DQN', mode='load', case='D', training_timesteps=1e6, render=render)
    
    env, model, start_time_tests, log_dir, env_RC = \
        train_RL(algorithm='DQN_RC_bc', mode='load', case='D', training_timesteps=0, 
                 render=render, expert_traj=os.path.join('trajectories','expert_traj_disc_28.npz'), 
                 return_RC=True, from_model='last_model')
    
    warmup_period_test  = 7*24*3600
    episode_length_test = 14*24*3600
    save_to_file = True
    
    # start learning from first day
    model.batch_size = 1*96
    # First time is (1382400 -2097000)/3600/24
    # First time is (1382400 -2097000)/3600/24
    test_peak(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, save_to_file, plot, env_RC)
    # test_typi(env, model, start_time_tests, episode_length_test, warmup_period_test, log_dir, save_to_file, plot, env_RC)
    
