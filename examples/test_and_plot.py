'''
Common functionality to test and plot an agent

'''

import requests
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from gym.core import Wrapper
import json
from matplotlib.pyplot import axis

def test_agent(env, model, start_time, episode_length, warmup_period,
               kpis_to_file=False, plot=False):
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
        if start_time==2678400:
            ckey='feb'
        elif start_time==26265600:
            ckey='nov'
        else:
            ckey=str(start_time)
    
        with open('kpis_{}.json'.format(ckey), 'w') as f:
            json.dump(kpis, f)
    
    if plot:
        plot_results(env, rewards)
    
    return observations, actions, rewards, kpis

def plot_results(env, rewards):
    df_res = pd.DataFrame()
    for point in list(env.all_measurement_vars.keys()) + list(env.all_input_vars.keys()):
        # Retrieve all simlation data
        res = requests.put('{0}/results'.format(env.url), data={'point_name':point,'start_time':0, 'final_time':3.1536e7}).json()
        df_res = pd.concat((df_res,pd.DataFrame(data=res[point], index=res['time'],columns=[point])), axis=1)
    df_res.index.name = 'time'
    df_res.reset_index(inplace=True)
        
    # Retrieve boundary condition data. 
    # Only way we have is through the forecast request. Take 10 points per step:
    env.reset()
    forecast_parameters = {'horizon':env.max_episode_length, 'interval':env.step_period/10}
    requests.put('{0}/forecast_parameters'.format(env.url),
                 data=forecast_parameters)
    forecast = requests.get('{0}/forecast'.format(env.url)).json()
    
    df_for = pd.DataFrame(forecast)
    
    df = pd.concat((reindex(df_res),reindex(df_for)), axis=1)
    df.dropna(axis=1,inplace=True) # there is one time column with nans...

    df = create_datetime(df)
    
    _, axs = plt.subplots(5, sharex=True, figsize=(8,6))
    x_time = df.index.to_pydatetime()

    axs[0].plot(x_time, df['reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
    axs[0].plot(x_time, df['TSetLow[1]'] -273.15, color='gray',         linewidth=1, label='Comfort setp.')
    axs[0].plot(x_time, df['TSetUpp[1]'] -273.15, color='gray',         linewidth=1, label='_nolegend_')
    axs[0].set_yticks(np.arange(15, 31, 5))
    axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
    
    axt = axs[0].twinx()
    axt.plot(x_time, df['pri'], color='dimgray', linestyle='dotted', linewidth=1, label='Price')
    axs[0].plot([],[], color='dimgray', linestyle='-', linewidth=1, label='Price')
    
    axt.set_ylim(0,0.3)
    axt.set_yticks(np.arange(0, 0.31, 0.1))
    axt.set_ylabel('(EUR/kWh)')   
    axt.set_ylabel('Price\n(EUR/kWh)')
    
    axs[1].plot(x_time, df['reaHeaPumY_y'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
    axs[1].set_ylabel('Heat pump\nmodulation\nsignal\n( - )')
    
    axs[2].plot(x_time, df['tdis_tot'], color='darkorange',   linestyle='-',   linewidth=1, label='_nolegend_')
    axs[2].set_ylabel('Thermal\ndiscomfort\n($Kh$)')
    
    axs[3].plot(x_time, df['cost_tot'], color='darkorange',   linestyle='-',   linewidth=1, label='_nolegend_')
    axs[3].set_yticks(np.arange(0, 151, 50))
    axs[3].set_ylabel('Operational\ncost\n(EUR)')
    
    axs[4].plot(x_time, df['TDryBul'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
    axs[4].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
    axs[4].set_yticks(np.arange(-5, 16, 5))
    axt = axs[4].twinx()
    
    axt.plot(x_time, df['HGloHor'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
    axt.set_ylabel('Solar\nirradiation\n($W$)')
    
    
    axs[4].plot([],[], color='deepskyblue', linestyle='-', linewidth=1, label='MPC, $T_h=12h$')
    axs[4].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='MPC, $T_h=24h$')
    axs[4].plot([],[], color='green',       linestyle='-', linewidth=1, label='Baseline')
    axs[4].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
    axs[4].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
    axs[4].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
    axs[4].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
    
    axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.tight_layout()
    
    plt.savefig('results_sim.pdf', bbox_inches='tight')
    
    plt.show()  

    
def reindex(df, interval=60, start=None, stop=None):
    # Define the index. Make sure last point is included if 
    # possible. If interval is not an exact divisor of stop,
    # the closest possible point under stop will be the end 
    # point in order to keep interval unchanged among index.
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
    
    # Create a datetime index for the data
    datetime = []
    for t in df['time']:
        datetime.append(pd.Timestamp('2020/1/1') + pd.Timedelta(t,'s'))
    df['datetime'] = datetime
    df.set_index('datetime', inplace=True)
    
    return df
    
    