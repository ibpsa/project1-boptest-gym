'''
Common functionality to test and plot an agent

'''

import matplotlib.pyplot as plt
from scipy import interpolate
from gym.core import Wrapper
from collections import OrderedDict
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import json
import os

from pyfmi import load_fmu
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
               log_dir=os.getcwd(), kpis_to_file=False, plot=False, env_RC=None):
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
    print('Simulating...')
    while done is False:
        actions_rewards = OrderedDict()
        actions_observs = OrderedDict()
        actions_returns = OrderedDict()
        
        measurement = env.unwrapped.last_measurement
        initial_states = env.observer.observe(measurement)
        
        for k,v in stat_map.items():
            initial_states[v] = initial_states.pop(k)
                
        print('From Tzon: {}'.format(initial_states['mod.bui.zon.capZon.TSta']-273.15))
        for a in range(0,11,5):
            actions_observs[a], actions_rewards[a] = env_RC.imagine(initial_states, np.array(a)) 
            _, q_values = model.predict(actions_observs[a], deterministic=True)
            cost_to_go = model.gamma*np.max(q_values) 
            actions_returns[a] = actions_rewards[a] + cost_to_go
            print('Action: {0}. Tzon: {1}. Reward: {2}. Cost-to-go: {3} Return: {4}'.format(a, 
                                                                                            env.observation_inverse(actions_observs[a])[1]-273.15, 
                                                                                            actions_rewards[a],
                                                                                            cost_to_go,
                                                                                            actions_returns[a]  ))
        
        # Find the action leading to the maximum return max(actions_returns.values())
        action = max(actions_returns, key=actions_returns.get)
        actrew = max(actions_rewards, key=actions_rewards.get)
        
        print('ACTION TAKEN IS: {}-----------------------'.format(action))
        
        if action != actrew:
            print('ACTION THAT HAD BEST REWARD WAS: {}-----------------------'.format(actrew))
            print('Successsss!!!!')
            
            
        # Advance the actual environment and store actual obs and rewards
        obs, reward, done, _ = env.step(np.asarray(action))    
        observations.append(obs)
        rewards.append(reward)
        # Advance the simple environment
        env_RC.advance_time_only()
        
    kpis = env.get_kpis()
    
    if kpis_to_file:
        with open(os.path.join(log_dir, 'kpis_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(kpis, f)
    
    if True:
        plot_results(env, rewards, plot_to_file=True, res_to_csv=True)
    
    # Back to random start time, just in case we're testing in the loop
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time = True
    else:
        env.random_start_time = True
    
    return observations, actions, rewards, kpis

def plot_results(env, rewards, points=['reaTZon_y','reaHeaPumY_y'],
                 log_dir=os.getcwd(), plot_to_file=False, res_to_csv=False):
    
    df_res = pd.DataFrame()
    if points is None:
        points = list(env.all_measurement_vars.keys()) + \
                 list(env.all_input_vars.keys())
        
    for point in points:
        # Retrieve all simulation data
        # We use env.start_time+1 to ensure that we don't return the last 
        # point from the initialization period to don't confuse it with 
        # actions taken by the agent
        res = requests.put('{0}/results'.format(env.url), 
                           data={'point_name':point,
                                 'start_time':env.start_time+1, 
                                 'final_time':3.1536e7}).json()
        df_res = pd.concat((df_res,pd.DataFrame(data=res[point], 
                                                index=res['time'],
                                                columns=[point])), axis=1)
        
    df_res.index.name = 'time'
    df_res.reset_index(inplace=True)
    df_res = reindex(df_res)
    
    # Retrieve boundary condition data. 
    # Only way we have is through the forecast request. 
    requests.put('{0}/initialize'.format(env.url), 
                 data={'start_time':df_res['time'].iloc[0],
                       'warmup_period':0}).json()
    # Store original forecast parameters
    forecast_parameters_original = requests.get('{0}/forecast_parameters'.format(env.url)).json()
    # Set forecast parameters for test. Take 10 points per step. 
    forecast_parameters = {'horizon':env.max_episode_length, 
                           'interval':env.step_period/10}
    requests.put('{0}/forecast_parameters'.format(env.url),
                 data=forecast_parameters)
    forecast = requests.get('{0}/forecast'.format(env.url)).json()
    # Back to original parameters, just in case we're testing during training
    requests.put('{0}/forecast_parameters'.format(env.url),
                 data=forecast_parameters_original)
        
    df_for = pd.DataFrame(forecast)
    df_for = reindex(df_for)
    df_for.drop('time', axis=1, inplace=True)
    
    df = pd.concat((df_res,df_for), axis=1)

    df = create_datetime(df)
    
    df.dropna(axis=0, inplace=True)
    
    if res_to_csv:
        df.to_csv(os.path.join(log_dir, 
                  'results_sim_{}.csv'.format(str(int(res['time'][0]/3600/24)))))
    
    rewards_time_days = np.arange(df_res['time'].iloc[0], 
                                  env.start_time+env.max_episode_length,
                                  env.step_period)/3600./24.
    f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                             fill_value='extrapolate')
    res_time_days = np.array(df['time'])/3600./24.
    rewards_reindexed = f(res_time_days)
    
    if not plt.get_fignums():
        # no window(s) open
        # fig = plt.figure(figsize=(10,8))
        _, axs = plt.subplots(4, sharex=True, figsize=(8,6))
    else:
        # get current figure. Combine this with plt.ion(), plt.figure()
        fig = plt.gcf()
        axs = fig.subplots(nrows=4, ncols=1, sharex=True)
            
    x_time = df.index.to_pydatetime()

    axs[0].plot(x_time, df['reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
    axs[0].plot(x_time, df['LowerSetp[1]'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
    axs[0].plot(x_time, df['UpperSetp[1]'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
    axs[0].set_yticks(np.arange(15, 31, 5))
    axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
    
    axt = axs[0].twinx()
    axt.plot(x_time, df['PriceElectricPowerHighlyDynamic'], color='dimgray', linestyle='dotted', linewidth=1, label='Price')
    axs[0].plot([],[], color='dimgray', linestyle='-', linewidth=1, label='Price')
    
    axt.set_ylim(0,0.3)
    axt.set_yticks(np.arange(0, 0.31, 0.1))
    axt.set_ylabel('(EUR/kWh)')   
    axt.set_ylabel('Price\n(EUR/kWh)')
    
    axs[1].plot(x_time, df['reaHeaPumY_y'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
    axs[1].set_ylabel('Heat pump\nmodulation\nsignal\n( - )')
    
    axs[2].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
    axs[2].set_ylabel('Rewards\n(-)')
    
    axs[3].plot(x_time, df['TDryBul'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
    axs[3].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
    axs[3].set_yticks(np.arange(-5, 16, 5))
    axt = axs[3].twinx()
    
    axt.plot(x_time, df['HDirNor'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
    axt.set_ylabel('Solar\nirradiation\n($W$)')
    
    axs[3].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='RL')
    axs[3].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
    axs[3].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
    axs[3].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
    axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
    
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.tight_layout()
    
    if plot_to_file:
        plt.savefig(os.path.join(log_dir, 
                    'results_sim_{}.pdf'.format(str(int(res['time'][0]/3600/24)))), 
                    bbox_inches='tight')
    
    plt.pause(0.001)
    plt.show()  

    
def reindex(df, interval=60, start=None, stop=None):
    '''
    Define the index. Make sure last point is included if 
    possible. If interval is not an exact divisor of stop,
    the closest possible point under stop will be the end 
    point in order to keep interval unchanged among index.
    
    ''' 
    
    if start is None:
        start = df['time'][df.index[0]]
    if stop is None:
        stop  = df['time'][df.index[-1]]  
    index = np.arange(start,stop+0.1,interval).astype(int)
    df_reindexed = df.reindex(index)
    
    # Avoid duplicates from FMU simulation. Duplicates lead to 
    # extrapolation errors
    df.drop_duplicates('time',inplace=True)
    
    for key in df_reindexed.keys():
        # Use linear interpolation 
        f = interpolate.interp1d(df['time'], df[key], kind='linear',
                                 fill_value='extrapolate')
        df_reindexed.loc[:,key] = f(index)
        
    return df_reindexed


def create_datetime(df):
    '''
    Create a datetime index for the data
    
    '''
    
    datetime = []
    for t in df['time']:
        datetime.append(pd.Timestamp('2020/1/1') + pd.Timedelta(t,'s'))
    df['datetime'] = datetime
    df.set_index('datetime', inplace=True)
    
    return df
    
    