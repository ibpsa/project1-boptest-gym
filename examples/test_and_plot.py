'''
Common functionality to test and plot an agent

'''

import requests
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from gym.core import Wrapper
import json
import pandas as pd
import copy

def test_agent(env, model, start_time, episode_length, warmup_period,
               N=3, plot=False):
    ''' Test model agent in env.
    Notice that there it is not possible to implement deterministic testing
    even when setting `deterministic=True`. 
    See e.g. 
    https://github.com/hill-a/stable-baselines/pull/492
    or
    https://github.com/hill-a/stable-baselines/issues/145
    or 
    https://openlab-flowers.inria.fr/t/how-many-random-seeds-should-i-use-statistical-power-analysis-in-deep-reinforcement-learning-experiments/457
    or
    https://datascience.stackexchange.com/questions/56308/why-do-trained-rl-agents-still-display-stochastic-exploratory-behavior-on-test
    
    Summary:
    pseudo-random seeds are set at INITIALIZATION of the model. 
    `deterministic=True` stops the agent from following the model, and 
    typically will select the mode (highest probability density) 
    of the action distribution. Therefore, setting `deterministic=True`
    is the right way to test RL algos since they'll act in a greedy way. 
    However, even with `deterministic=True`, the agent is still stochastic 
    and dependent on the tensorflow seeds that are set at initialization 
    and that CANNOT BE CONTROLLED. Reload is therefore not supported to 
    compute deterministic cases. Only real deterministic cases are 
    achieved when running several cases for the same tensorflow session, 
    i.e. when initializing the RL model only once and performing several 
    tests on it. 
    
    '''
    
    if N==1:
        deterministic=True
    else:
        # The only way to perform different simulation runs is with 
        # `deterministic=False`, notice though that the RL algo is not 
        # greedy in these cases. 
        deterministic=False
    
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
    
    res_list = []
    rewards_list = []
    kpis_list = []
    # Compute the mean and the 68% confidence interval of N simulation runs 
    for i in range(N):
        print('Starting simulation run number {}'.format(i))
        # Reset environment
        obs = env.reset()
        
        # Simulation loop
        done = False
        observations = [obs]
        actions = []
        rewards = []
        print('Simulating...')
        
        # The following is an attempt to change the random seed for each
        # simulation run but I'm afraid it has no effect for some reason, 
        # at least when `deterministic=True`
        model.set_random_seed(i*123456)
        
        while done is False:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
        
        kpis = env.get_kpis()
        
        if start_time==2678400:
            ckey='feb'
        elif start_time==26265600:
            ckey='nov'
        else:
            ckey=str(start_time)
        
        with open('kpis_{0}_{1}.json'.format(ckey,i+1), 'w') as f:
            json.dump(kpis, f)
        
        res = requests.get('{0}/results'.format(env.url)).json()
        res_list.append(res)
        rewards_list.append(rewards)
        kpis_list.append(kpis)
    
    kpis_df = pd.DataFrame(kpis_list)
    kpis_avg = kpis_df.mean()
    kpis_std = kpis_df.std()
    with open('kpis_{0}_avg.json'.format(ckey), 'w') as f:
        json.dump(kpis_avg.to_dict(), f)
    with open('kpis_{0}_std.json'.format(ckey), 'w') as f:
        json.dump(kpis_std.to_dict(), f)
        
    if plot:
        env.reset()
        # Retrieve boundary condition data only once. 
        # Only way we have is through the forecast request. Take 10 points per step:
        forecast_parameters = {'horizon':env.max_episode_length, 'interval':env.Ts/10}
        requests.put('{0}/forecast_parameters'.format(env.url),
                     data=forecast_parameters)
        forecast = requests.get('{0}/forecast'.format(env.url)).json()
        # Do not mess up forecasting time with simulation time
        forecast['time_forecast'] = forecast.pop('time')
        plot_results(env, forecast, res_list, rewards_list)
        
    return observations, actions, rewards, kpis

def plot_results(env, forecast, res_list, rewards_list):
    
    _, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    
    meas = 'reaTZon_y' # Measurement
    cInp = 'reaHeaPumY_y' # Control input
    
    if env.scenario['electricity_price'] == 'constant':
        price_name = 'PriceElectricPowerConstant'
    elif env.scenario['electricity_price'] == 'dynamic':
        price_name = 'PriceElectricPowerDynamic'
    elif env.scenario['electricity_price'] == 'highly_dynamic':
        price_name = 'PriceElectricPowerHighlyDynamic'
    
    forecast_time = np.array(forecast['time_forecast'])/3600./24.
    res_pric = np.array(forecast[price_name])
    
    axs[0].set_ylabel('Zone temperature\n($^\circ$C)')
    axs[1].set_ylabel('Heat pump\nmodulating signal\n(-)')
    twin = axs[1].twinx()
    twin.plot(forecast_time, res_pric, 'grey', linewidth=1, label='Price')
    twin.set_ylabel('(EUR/kWh)')
    axs[2].set_ylabel('Rewards\n(-)')
    
    res_time_days=None
    res_meas_df=None
    res_cInp_df=None
    res_rewards_df=None
    for i in range(len(res_list)):
        res = res_list[i]
        rewards = rewards_list[i]        
        res_all = {}
        res_all.update(res['u'])
        res_all.update(res['y'])
        
        # Plot boundary condition data in axs[0] 
        if res_time_days is None:
            res_time_days = np.array(res_all['time'])/3600./24.
            res_lSet = np.array(res_all['reaTSetHea_y'])
            res_uSet = np.array(res_all['reaTSetCoo_y'])
            axs[0].plot(res_time_days, res_lSet-273.15)
            axs[0].plot(res_time_days, res_uSet-273.15)
        
        # Measurements
        f = interpolate.interp1d(np.array(res_all['time'])/3600./24., res_all[meas], kind='linear',
                                 fill_value='extrapolate')
        meas_reindexed = f(res_time_days)        
        
        # Control inputs
        f = interpolate.interp1d(np.array(res_all['time'])/3600./24., res_all[cInp], kind='linear',
                                 fill_value='extrapolate')
        cInp_reindexed = f(res_time_days) 
        
        # Rewards
        rewards_time_days = np.arange(env.start_time, 
                                      env.start_time+env.max_episode_length,
                                      env.Ts)/3600./24.
        f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                                 fill_value='extrapolate')
        rewards_reindexed = f(res_time_days)
        
        # Measurements
        if res_meas_df is None:
            res_meas_df = pd.DataFrame(meas_reindexed,index=res_time_days,columns=[meas+'_{}'.format(i)])
        else:
            df = pd.DataFrame(meas_reindexed,index=res_time_days,columns=[meas+'_{}'.format(i)])
            res_meas_df = pd.concat([res_meas_df, df], axis=1)
        
        # Control inputs
        if res_cInp_df is None:
            res_cInp_df = pd.DataFrame(cInp_reindexed,index=res_time_days,columns=[cInp+'_{}'.format(i)])
        else:
            df = pd.DataFrame(cInp_reindexed,index=res_time_days,columns=[cInp+'_{}'.format(i)])
            res_cInp_df = pd.concat([res_cInp_df, df], axis=1)
        
        # Rewards
        if res_rewards_df is None:
            res_rewards_df = pd.DataFrame(rewards_reindexed,index=res_time_days,columns=['rewards_{}'.format(i)])
        else:
            df = pd.DataFrame(rewards_reindexed,index=res_time_days,columns=['rewards_{}'.format(i)])
            res_rewards_df = pd.concat([res_rewards_df, df], axis=1)
    
    # Measurements
    res_meas_df_old = copy.deepcopy(res_meas_df)
    res_meas_df[meas+'_avg'] = np.array(res_meas_df_old.mean(axis=1))
    res_meas_df[meas+'_std'] = np.array(res_meas_df_old.std(axis=1))
    
    # Control inputs
    res_cInp_df_old = copy.deepcopy(res_cInp_df)
    res_cInp_df[meas+'_avg'] = np.array(res_cInp_df_old.mean(axis=1))
    res_cInp_df[meas+'_std'] = np.array(res_cInp_df_old.std(axis=1))
    
    # Rewards
    res_rewards_df_old = copy.deepcopy(res_rewards_df)
    res_rewards_df[meas+'_avg'] = np.array(res_rewards_df_old.mean(axis=1))
    res_rewards_df[meas+'_std'] = np.array(res_rewards_df_old.std(axis=1))
    
    # Measurements
    axs[0].plot(res_time_days, res_meas_df[meas+'_avg']-273.15, 'b-',label=meas)
    axs[0].fill_between(res_time_days, 
     res_meas_df[meas+'_avg'] - 273.15 - res_meas_df[meas+'_std'], 
     res_meas_df[meas+'_avg'] - 273.15 + res_meas_df[meas+'_std'],
     color='b', alpha=.1)
    
    # Control inputs
    axs[1].plot(res_time_days, res_cInp_df[meas+'_avg'], 'b-',label=cInp)
    axs[1].fill_between(res_time_days, 
     res_cInp_df[meas+'_avg'] - res_cInp_df[meas+'_std'], 
     res_cInp_df[meas+'_avg'] + res_cInp_df[meas+'_std'],
     color='b', alpha=.1)
    
    # Rewards
    axs[2].plot(res_time_days, res_rewards_df[meas+'_avg'], 'b-',label=cInp)
    axs[2].fill_between(res_time_days, 
     res_rewards_df[meas+'_avg'] - res_rewards_df[meas+'_std'], 
     res_rewards_df[meas+'_avg'] + res_rewards_df[meas+'_std'],
     color='b', alpha=.1)
    
    axs[2].set_xlabel('Day of the year')
    plt.show()   
    