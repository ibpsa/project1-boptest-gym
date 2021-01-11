'''
Common functionality to test and plot an agent

'''

import requests
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from gym.core import Wrapper
import json

def test_agent(env, model, start_time, episode_length, warmup_period,
               plot=False):
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
    
    if start_time==2678400:
        ckey='feb'
    elif start_time==26265600:
        ckey='nov'
    
    with open('kpis_{}.json'.format(ckey), 'w') as f:
        json.dump(kpis, f)
    
    if plot:
        plot_results(env, rewards)
    
    return observations, actions, rewards, kpis

def plot_results(env, rewards):
    res = requests.get('{0}/results'.format(env.url)).json()
    # Retrieve boundary condition data. 
    # Only way we have is through the forecast request. Take 10 points per step:
    env.reset()
    forecast_parameters = {'horizon':env.max_episode_length, 'interval':env.Ts/10}
    requests.put('{0}/forecast_parameters'.format(env.url),
                 data=forecast_parameters)
    forecast = requests.get('{0}/forecast'.format(env.url)).json()
    res_all = {}
    res_all.update(res['u'])
    res_all.update(res['y'])
    # Do not mess up forecasting time with simulation time
    forecast['time_forecast'] = forecast.pop('time')
    res_all.update(forecast)
    
    _ = plt.figure(figsize=(10,8))
    
    meas_names = ['reaTZon_y'] # measurements
    cInp_names = ['reaHeaPumY_y'] # control inputs
    if env.scenario['electricity_price'] == 'constant':
        price_name = 'PriceElectricPowerConstant'
    elif env.scenario['electricity_price'] == 'dynamic':
        price_name = 'PriceElectricPowerDynamic'
    elif env.scenario['electricity_price'] == 'highly_dynamic':
        price_name = 'PriceElectricPowerHighlyDynamic'
    
    res_time_days = np.array(res_all['time'])/3600./24.
    forecast_time = np.array(forecast['time_forecast'])/3600./24.
    res_lSet = np.array(res_all['reaTSetHea_y'])
    res_uSet = np.array(res_all['reaTSetCoo_y'])
    res_pric = np.array(res_all[price_name])
    res_meas = {meas: np.array(res_all[meas]) for meas in meas_names}
    res_cInp = {cInp: np.array(res_all[cInp]) for cInp in cInp_names}

    ax1 = plt.subplot(3, 1, 1)
    for meas in res_meas.keys():
        plt.plot(res_time_days, res_meas[meas]-273.15, label=meas)
        
    plt.plot(res_time_days, res_lSet-273.15)
    plt.plot(res_time_days, res_uSet-273.15)
    plt.legend()
    ax1.set_ylabel('Zone temperature\n($^\circ$C)')
    
    ax2 = plt.subplot(3, 1, 2)
    for cInp in res_cInp.keys():
        plt.plot(res_time_days, res_cInp[cInp], label=cInp)
    ax2.set_ylabel('Heat pump\nmodulating signal\n(-)')
    ax2twin = ax2.twinx()
    ax2twin.plot(forecast_time, res_pric, 'grey', linewidth=1, label='Price')
    ax2twin.set_ylabel('(EUR/kWh)')
      
    rewards_time_days = np.arange(env.start_time, 
                                  env.start_time+env.max_episode_length,
                                  env.Ts)/3600./24.
    f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                             fill_value='extrapolate')
    rewards_reindexed = f(res_time_days)
    
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(res_time_days, rewards_reindexed, label='rewards')
    ax3.set_ylabel('Rewards\n(-)')
    
    plt.show()   
    
    