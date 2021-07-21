'''
Common functionality to test and plot an agent

'''

import matplotlib.pyplot as plt
from gym.core import Wrapper
from collections import OrderedDict
import numpy as np
import pandas as pd
import json
import os

from pyfmi import load_fmu
from examples.test_and_plot import plot_results
from state_estimator.observer_UKF import Observer_UKF

# Set simulation inputs and measurements
meas_map={'zon.capZon.heaPor.T':    'reaTZon_y'}
cInp_map={'yHeaPum':                'oveHeaPumY_u'}
dist_map={'TAmb':'weaSta_reaWeaTDryBul_y', 
          'irr[1]':'weaSta_reaWeaHDirNor_y', 
          #'intGai[1]':'InternalGainsRad[1]', 
          #'intGai[2]':'InternalGainsCon[1]', 
          #'intGai[3]':'InternalGainsLat[1]',
          #'TSetLow[1]':'LowerSetp[1]',
          #'TSetUpp[1]':'UpperSetp[1]',
          #'pri':scenario_pars[scenario['electricity_price']]
          } 
stat_map = {}
stat_map['hea.capFlo.heaPor.T'] = 'mod.bui.hea.capFlo.TSta'
stat_map['zon.capInt.heaPor.T'] = 'mod.bui.zon.capInt.TSta'
stat_map['zon.capEmb.heaPor.T'] = 'mod.bui.zon.capEmb.TSta'
stat_map['zon.capWal.heaPor.T'] = 'mod.bui.zon.capWal.TSta'
stat_map['zon.capZon.heaPor.T'] = 'mod.bui.zon.capZon.TSta'
    
# Define covariance measurement noise
cov_meas_noise = 0.

# Set scenario parameters for the simulation
days_json = {"peak_heat_day": 23, "typical_heat_day": 115}
year_bgn = pd.Timestamp('20210101 00:00:00')

scenario_pars = {}
scenario_pars['training'] = {}
scenario_pars['training']['bgn_sim_time'] = '20210201 00:00:00' 
scenario_pars['training']['end_sim_time'] = '20210401 00:00:00'
scenario_pars['peak_heat_day'] = {}
scenario_pars['peak_heat_day']['bgn_sim_time'] = '20210117 00:00:00' 
scenario_pars['peak_heat_day']['end_sim_time'] = '20210131 00:00:00' 
scenario_pars['typical_heat_day'] = {}
scenario_pars['typical_heat_day']['bgn_sim_time'] = '20210419 00:00:00' 
scenario_pars['typical_heat_day']['end_sim_time'] = '20210503 00:00:00'

scenario_pars['constant'] = 'PriceElectricPowerConstant'
scenario_pars['dynamic'] = 'PriceElectricPowerDynamic'
scenario_pars['highly_dynamic'] = 'PriceElectricPowerHighlyDynamic'

scenario = {'electricity_price':'PriceElectricPowerHighlyDynamic',
            'time_period':'peak_heat_day'}

def test_agent(env, model, start_time, episode_length, warmup_period,
               log_dir=os.getcwd(), save_to_file=False, plot=False, env_RC=None):
    ''' Test model agent in env.
    
    '''
    
    if start_time == (days_json['peak_heat_day']-7)*24*3600:
        scenario['time_period'] = 'peak_heat_day'
    elif start_time == (days_json['typical_heat_day']-7)*24*3600:
        scenario['time_period'] = 'typical_heat_day'
    else:
        raise KeyError('start_time does not agree with any scenario period') 
    
    # Find datetime for starting simulation time
    time_sim_0 = pd.Timestamp(scenario_pars[scenario['time_period']]['bgn_sim_time'])
    
    # Set a fixed start time
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time   = False
        env.unwrapped.start_time          = start_time
        env.unwrapped.max_episode_length  = episode_length
        env.unwrapped.warmup_period       = warmup_period
    else:
        env.random_start_time   = False
        env.start_time          = start_time
        env.max_episode_length  = episode_length
        env.warmup_period       = warmup_period
    
    # Reset environment
    obs = env.reset()
    
    if env_RC is not None:
        if isinstance(env_RC,Wrapper): 
            env_RC.unwrapped.random_start_time   = False
            env_RC.unwrapped.start_time          = start_time
            env_RC.unwrapped.max_episode_length  = episode_length
            env_RC.unwrapped.warmup_period       = warmup_period
        else:
            env_RC.random_start_time   = False
            env_RC.start_time          = start_time
            env_RC.max_episode_length  = episode_length
            env_RC.warmup_period       = warmup_period
        
        # env_RC.url_regr             = 'http://127.0.0.1:5000'
        env_RC.unwrapped.url_regr   = 'http://127.0.0.1:5000'
        
        # Reset environment
        _ = env_RC.reset()
    
    #=================================================================
    # STATE OBSERVER
    #=================================================================

    # Load model for state observer
    # I obtained this model with OpenModelica
    fmu_path='TestCaseOptimization.fmu'
    model_ukf = load_fmu(fmu_path, enable_logging=False, 
                         log_file_name='logUkfLoad.txt', log_level=7)
    
    # Load the initial states as calculated with the initialization data 
    initial_state = pd.read_csv('initial_state_{}.csv'.format(scenario['time_period']),
                                index_col=0)
    initial_state.index = [time_sim_0]
    
    # Instantiate observer
    env.unwrapped.meas_map   = meas_map
    env.unwrapped.cInp_map   = cInp_map
    env.unwrapped.dist_map   = dist_map
    env.unwrapped.meas_names = meas_map.keys()
    env.unwrapped.cInp_names = cInp_map.keys()
    env.unwrapped.dist_names = dist_map.keys()
    env.unwrapped.stat_names = initial_state.columns
    env.unwrapped.time_sim   = [time_sim_0]
    env.unwrapped.Ts         = env.step_period
    env.unwrapped.pars_fixed = None
    
    env.observer  = Observer_UKF(parent=env.unwrapped, model_ukf = model_ukf, 
                                 cov_meas_noise = cov_meas_noise,
                                 stai = initial_state, pars_json_file='ZonWalIntEmb_B_TConTEva_C1.json')      
    
    # Define a more accurate set of covariance matrices based on training data
    # The specific numbers for P_v are obtained from `bb_load_models` when launching the
    # simulation for 14 days and sampling of 900 s (training conditions)
    # The specific numbers for P_n are just estimations based on how the measurements 
    # are obtained (the power is by far much less reliable than the temperature measurments)
    P_n = {}
    P_v = {} 
    P_0 = {} 
    
    P_n['zon.capZon.heaPor.T']         = cov_meas_noise
    
    P_v['zon.capZon.heaPor.T']         = 0.17764571883198466**2. # As the RMSE obtained for system identification during the training period. Calculated in load_gb_mod 
    P_v['zon.capWal.heaPor.T']         = P_v['zon.capZon.heaPor.T']*1.1 # This one is calculated as a 10 percent more compared to the empirically obtained process variance of the operational zone temperature
    P_v['zon.capInt.heaPor.T']         = P_v['zon.capZon.heaPor.T']*1.1 # This one is calculated as a 10 percent more compared to the empirically obtained process variance of the operational zone temperature
    P_v['zon.capEmb.heaPor.T']         = P_v['zon.capZon.heaPor.T']*1.1 # This one is calculated as a 10 percent more compared to the empirically obtained process variance of the operational zone temperature
    P_v['hea.capFlo.heaPor.T']         = P_v['zon.capZon.heaPor.T']*1.1 # This one is calculated as a 10 percent more compared to the empirically obtained process variance of the operational zone temperature
    
    P_0['zon.capZon.heaPor.T']         = P_v['zon.capZon.heaPor.T'] + P_n['zon.capZon.heaPor.T']
    P_0['zon.capWal.heaPor.T']         = P_v['zon.capWal.heaPor.T']
    P_0['zon.capInt.heaPor.T']         = P_v['zon.capInt.heaPor.T']
    P_0['zon.capEmb.heaPor.T']         = P_v['zon.capEmb.heaPor.T']
    P_0['hea.capFlo.heaPor.T']         = P_v['hea.capFlo.heaPor.T']
    
    # Update the options of the ukf 
    env.observer.ukf.update_options(P_v=P_v, P_n=P_n, P_0=P_0)
    
    # Update confidence intervals of state observer accordingly
    for stat in env.observer.stat_names:
        env.observer.conf.loc[time_sim_0, stat] = np.sqrt(env.observer.ukf.options['P_0'][stat])
    
    #=================================================================
    # SIMULATION LOOP
    #=================================================================
    done = False
    observations = [obs]
    actions = []
    rewards = []
    initial_states_list  = []
    imagined_temper_list = []
    imagined_rewinm_list = []
    imagined_rewtog_list = []
    imagined_return_list = []
    print('Simulating...')
    env_RC.unwrapped.debug = False
    while done is False:
        imagined_rewinm = OrderedDict()
        imagined_temper = OrderedDict()
        imagined_rewtog = OrderedDict()
        imagined_observ = OrderedDict()
        imagined_return = OrderedDict()
        
        measurement = env.unwrapped.last_measurement
        initial_states = env.observer.observe(measurement)
        for k,v in stat_map.items():
            initial_states[v] = initial_states.pop(k)
        initial_states_list.append(initial_states)
        print('From Tzon: {}'.format(initial_states['mod.bui.zon.capZon.TSta']-273.15))
        for a in range(0,11,2):
            imagined_observ[a], imagined_rewinm[a] = env_RC.imagine(initial_states, np.array(a)) 
            imagined_temper[a] = env.observation_inverse(imagined_observ[a])[1]-273.15
            _, q_values = model.predict(imagined_observ[a], deterministic=True)
            imagined_rewtog[a] = model.gamma*np.max(q_values) 
            imagined_return[a] = imagined_rewinm[a] + imagined_rewtog[a]
            
            print('Action: {0}. Tzon: {1}. Inmediate Reward: {2}. Reward-to-go: {3} Return: {4}'.format(a, 
                imagined_temper[a], 
                imagined_rewinm[a],
                imagined_rewtog[a],
                imagined_return[a]  ))
        
        # Find the action leading to the maximum return max(imagined_return.values())
        action = max(imagined_return, key=imagined_return.get)
        actrew = max(imagined_rewinm, key=imagined_rewinm.get)
        
        print('ACTION TAKEN IS: {}-----------------------'.format(action))
        if action != actrew:
            print('ACTION THAT HAD BEST REWARD WAS: {}-----------------------'.format(actrew))
            print('Successsss!!!!')
            
        # Advance the actual environment and store actual obs and rewards
        obs, reward, done, _ = env.step(np.asarray(action))    
        observations.append(obs)
        rewards.append(reward)
        imagined_rewinm_list.append(imagined_rewinm)
        imagined_temper_list.append(imagined_temper)
        imagined_rewtog_list.append(imagined_rewtog)
        imagined_return_list.append(imagined_return)

        # Advance the simple environment
        env_RC.advance_time_only()
    
    #=================================================================
    # END OF SIMULATION LOOP
    #=================================================================
    kpis = env.get_kpis()
    
    if save_to_file:
        with open(os.path.join(log_dir, 'results_tests', 'kpis_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(kpis, f)
        with open(os.path.join(log_dir, 'results_tests', 'rewinm_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(imagined_rewinm_list, f)
        with open(os.path.join(log_dir, 'results_tests', 'temper_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(imagined_temper_list, f)
        with open(os.path.join(log_dir, 'results_tests', 'rewtog_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(imagined_rewtog_list, f)
        with open(os.path.join(log_dir, 'results_tests', 'rewret_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(imagined_return_list, f)
        with open(os.path.join(log_dir, 'results_tests', 'rewards_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(rewards, f)
        with open(os.path.join(log_dir, 'results_tests', 'inistates_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(initial_states_list, f)
                               
    if plot:
        plot_results(env, rewards, save_to_file=save_to_file, log_dir=log_dir)
    
    # Back to random start time, just in case we're testing in the loop
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time = True
    else:
        env.random_start_time = True
    
    return observations, actions, rewards, kpis

    