'''
Module to run the baseline controller of a model using the `BoptestGymEnv` 
interface. This example does not learn any policy, it is rather used 
to test the environment. 
The BOPTEST bestest_hydrinic_heat_pump case needs to be deployed.  

'''
import numpy as np
import requests
import matplotlib.pyplot as plt
from boptestGymEnv import BoptestGymEnv
from scipy import interpolate


url = 'http://127.0.0.1:5000'

def run_reward_default(plot=False):
    '''Run example with default reward function. 
    
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.
    
    Returns
    -------
    rewards : list
        Rewards obtained in simulation
        
    
    '''
    
    rewards = run(envClass=BoptestGymEnv, plot=plot)
        
    return rewards


def run_reward_custom(plot=False):
    '''Run example with customized reward function. 
    
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.
    
    Returns
    -------
    rewards : list
        Rewards obtained in simulation
        
    
    '''
    
    # Define a parent class as a wrapper to override the reward function
    class BoptestGymEnvCustom(BoptestGymEnv):
        
        def compute_reward(self):
            '''Custom reward function that penalizes less the discomfort
            and thus more the operational cost.
            
            '''
            
            # Define relative weight for discomfort 
            w = 0.1
            
            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi'.format(self.url)).json()
            
            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot'] + w*kpis['tdis_tot']
            
            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)
            
            self.objective_integrand = objective_integrand
            
            return reward
    
    rewards = run(envClass=BoptestGymEnvCustom, plot=plot)
        
    return rewards
    
def run(envClass, plot=False):
    # Use the first 3 days of February for testing with 3 days for initialization
    env = envClass(url                 = url,
                   observations        = ['reaTZon_y'], 
                   lower_obs_bounds    = [280.],
                   upper_obs_bounds    = [310.],
                   random_start_time   = False,
                   start_time          = 31*24*3600,
                   episode_length      = 3*24*3600,
                   warmup_period       = 3*24*3600,
                   Ts                  = 3600)
    
    # Define an empty action list to don't overwrite any input
    env.actions = [] 
    
    # Reset environment
    _ = env.reset()
    
    # Simulation loop
    done = False
    rewards = []
    print('Simulating...')
    while done is False:
        _, reward, done, _ = env.step([])
        rewards.append(reward)
        
    if plot:
        plot_results(env, rewards)
        
    return rewards

def plot_results(env, rewards):
    res = requests.get('{0}/results'.format(env.url)).json()
    res_all = {}
    res_all.update(res['u'])
    res_all.update(res['y'])

    _ = plt.figure(figsize=(10,8))
    
    meas_names = ['reaTZon_y'] # measurements
    cInp_names = ['reaHeaPumY_y'] # control inputs

    res_time_days = np.array(res_all['time'])/3600./24.
    res_lSet = np.array(res_all['reaTSetHea_y'])
    res_uSet = np.array(res_all['reaTSetCoo_y'])
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

    rewards_time_days = np.arange(env.start_time, 
                                  env.start_time+env.episode_length,
                                  env.Ts)/3600./24.
    f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                             fill_value='extrapolate')
    rewards_reindexed = f(res_time_days)
    
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(res_time_days, rewards_reindexed, label='rewards')
    ax3.set_ylabel('Rewards\n(-)')
    
    plt.show()   
        
if __name__ == "__main__":

    rewards = run_reward_custom(plot=True)
    