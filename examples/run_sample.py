'''
Module to run a random controller using the `BoptestGymEnv` 
interface. This example does not learn any policy, it is rather used 
to test the environment. 
The BOPTEST bestest_hydrinic_heat_pump case needs to be deployed.  

'''

from boptestGymEnv import BoptestGymEnv, NormalizedActionWrapper
from examples.test_and_plot import test_agent
import random

url = 'http://127.0.0.1'

# Seed for random starting times of episodes
random.seed(123456)

start_time_test     = 31*24*3600
episode_length_test = 3*24*3600
warmup_period_test  = 3*24*3600

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
    
def run(envClass, wrapper=None, plot=False):
    # Use the first 3 days of February for testing with 3 days for initialization
    env = envClass(url                 = url,
                   actions             = ['oveHeaPumY_u'],
                   observations        = {'reaTZon_y':(280.,310.)}, 
                   random_start_time   = False,
                   start_time          = 31*24*3600,
                   max_episode_length  = 3*24*3600,
                   warmup_period       = 3*24*3600,
                   step_period         = 3600)
    
    # Add wrapper if any
    if wrapper is not None:
        env = wrapper(env)
    
    model = SampleModel(env)
    
    # Perform test
    observations, actions, rewards, _ = test_agent(env, model, 
                         start_time=start_time_test, 
                         episode_length=episode_length_test,
                         warmup_period=warmup_period_test,
                         plot=plot)

    # stop the test
    env.stop()
    
    return observations, actions, rewards
        
class SampleModel(object):
    '''Dummy class that generates random actions. It therefore does not
    simulate the baseline controller, but is still maintained here because
    also serves as a simple case to test features. 
    
    '''
    def __init__(self, env):
        self.env = env
        # Seed for action space
        self.env.action_space.seed(123456)
    def predict(self,obs, deterministic=True):
        return self.env.action_space.sample(), obs
        
if __name__ == "__main__":
    rewards = run_normalized_action_wrapper(plot=True)
    