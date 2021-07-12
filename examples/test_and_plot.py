'''
Common functionality to test and plot an agent

'''

import matplotlib.pyplot as plt
from scipy import interpolate
from gym.core import Wrapper
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import json
import os


def test_agent(env, model, start_time, episode_length, warmup_period,
               log_dir=os.getcwd(), kpis_to_file=False, plot=False):
    ''' Test model agent in env.
    
    '''
        
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
    
    # Simulation loop
    done = False
    observations = [obs]
    actions = []
    rewards = []
    print('Simulating...')
    while done is False:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
    
    kpis = env.get_kpis()
    
    if kpis_to_file:
        with open(os.path.join(log_dir, 'kpis_{}.json'.format(str(int(start_time/3600/24)))), 'w') as f:
            json.dump(kpis, f)
    
    if plot:
        plot_results(env, rewards)
    
    # Back to random start time, just in case we're testing in the loop
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time = True
    else:
        env.random_start_time = True
    
    return observations, actions, rewards, kpis

def plot_results(env, rewards, points=['reaTZon_y','reaHeaPumY_y'],
                 log_dir=os.getcwd(), plot_to_file=False):
    
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
    
    