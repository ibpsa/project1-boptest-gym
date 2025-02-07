'''
Module to run the baseline controller of a model using the `BoptestGymEnv` 
interface. This example does not learn any policy, it is rather used 
to test the environment. 
The BOPTEST bestest_hydrinic_heat_pump case needs to be deployed.  

'''
import numpy as np
import requests
import random
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, NormalizedActionWrapper
from examples.test_and_plot import test_agent

url = 'http://127.0.0.1'

# Seed for random starting times of episodes
random.seed(123456)

start_time_test     = 31*24*3600
episode_length_test = 3*24*3600
warmup_period_test  = 3*24*3600

def run_reward_default(plot=False):
    '''Run example with default reward function. 
    
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.
    
    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation
    
    '''
    
    observations, actions, rewards = run(envClass=BoptestGymEnv, plot=plot)
        
    return observations, actions, rewards

def run_reward_custom(plot=False):
    '''Run example with customized reward function. 
    
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.
    
    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation
    
    '''
    
    # Define a parent class as a wrapper to override the reward function
    class BoptestGymEnvCustom(BoptestGymEnv):
        
        def get_reward(self):
            '''Custom reward function that penalizes less the discomfort
            and thus more the operational cost.
            
            '''
            
            # Define relative weight for discomfort 
            w = 0.1
            
            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
            
            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot']*12.*16. + w*kpis['tdis_tot']
            
            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)
            
            self.objective_integrand = objective_integrand
            
            return reward
    
    observations, actions, rewards = run(envClass=BoptestGymEnvCustom, plot=plot)
        
    return observations, actions, rewards

def run_reward_clipping(plot=False):
    '''Run example with clipped reward function. 
    
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation
    
    '''
    
    # Define a parent class as a wrapper to override the reward function
    class BoptestGymEnvClipping(BoptestGymEnv):
        
        def get_reward(self):
            '''Clipped reward function that has the value either -1 when
            there is any cost/discomfort, or 0 where there is not cost 
            nor discomfort. This would be the simplest reward to learn for
            an agent. 
            
            '''
            
            # Compute BOPTEST core kpis
            kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
            
            # Calculate objective integrand function at this point
            objective_integrand = kpis['cost_tot']*12.*16. + kpis['tdis_tot']
            
            # Compute reward
            reward = -(objective_integrand - self.objective_integrand)
            
            # Filter to be either -1 or 0
            reward = np.sign(reward)
            
            self.objective_integrand = objective_integrand
            
            return reward
    
    observations, actions, rewards = run(envClass=BoptestGymEnvClipping, plot=plot)
        
    return observations, actions, rewards

def run_normalized_observation_wrapper(plot=False):
    '''Run example with normalized observation wrapper. 
     
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.
     
    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation
        
    '''

    observations, actions, rewards = run(envClass=BoptestGymEnv, 
                  wrapper=NormalizedObservationWrapper,
                  plot=plot)
         
    return observations, actions, rewards

def run_normalized_action_wrapper(plot=False):
    '''Run example with normalized action wrapper. 
     
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.
     
    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation
        
    '''

    observations, actions, rewards = run(envClass=BoptestGymEnv, 
                  wrapper=NormalizedActionWrapper,
                  plot=plot)
         
    return observations, actions, rewards

def run_highly_dynamic_price(plot=False):
    '''Run example when setting the highly dynamic price scenario of BOPTEST. 
     
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.
     
    Returns
    -------
    observations : list
        Observations obtained in simulation
    actions : list
        Actions applied in simulation
    rewards : list
        Rewards obtained in simulation
        
    '''

    observations, actions, rewards = run(envClass=BoptestGymEnv, 
                  scenario={'electricity_price':'highly_dynamic'},
                  plot=plot)
         
    return observations, actions, rewards
    
def run(envClass, wrapper=None, scenario={'electricity_price':'constant'}, 
        plot=False):
    # Use the first 3 days of February for testing with 3 days for initialization
    env = envClass(url                 = url,
                   actions             = ['oveHeaPumY_u'],
                   observations        = {'reaTZon_y':(280.,310.)}, 
                   random_start_time   = False,
                   start_time          = 31*24*3600,
                   max_episode_length  = 3*24*3600,
                   warmup_period       = 3*24*3600,
                   scenario            = scenario,
                   step_period         = 3600)
    
    # Define an empty action list to don't overwrite any input
    env.actions = [] 
    
    # Add wrapper if any
    if wrapper is not None:
        env = wrapper(env)
    
    model = BaselineModel()
    # Perform test
    observations, actions, rewards, _ = test_agent(env, model, 
                         start_time=start_time_test, 
                         episode_length=episode_length_test,
                         warmup_period=warmup_period_test,
                         plot=plot)

    # stop the test
    env.stop()

    return observations, actions, rewards
        
class BaselineModel(object):
    '''Dummy class for baseline model. It simply returns empty list when 
    calling `predict` method. 
    
    '''
    def __init__(self):
        pass
    def predict(self, obs, deterministic=True):
        return [], obs
    
class SampleModel(object):
    '''Dummy class that generates random actions. It therefore does not
    simulate the baseline controller, but is still maintained here because
    also serves as a simple case to test features. 
    
    '''
    def __init__(self):
        pass
    def predict(self,obs, deterministic=True):
        return self.action_space.sample(), obs
        
if __name__ == "__main__":
    rewards = run_reward_custom(plot=True)
    