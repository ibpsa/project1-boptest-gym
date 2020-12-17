'''
Module to test features of the OpenAI-Gym interface for BOPTEST. 
The BOPTEST bestest_hydronic_heat_pump case needs to be deployed to perform
the tests. 

'''

import unittest
import utilities
import os
import pandas as pd
import random
from examples import run_baseline
from collections import OrderedDict
from boptestGymEnv import BoptestGymEnv
from stable_baselines.common.env_checker import check_env

url = 'http://127.0.0.1:5000'

class BoptestGymEnvTest(unittest.TestCase, utilities.partialChecks):
    '''Tests the OpenAI-Gym interface for BOPTESTS.
         
    '''
 
    def setUp(self):
        '''Setup for each test.
         
        '''
        self.env = BoptestGymEnv(url                 = url,
                                actions             = ['oveHeaPumY_u'],
                                observations        = ['reaTZon_y'], 
                                lower_obs_bounds    = [280.],
                                upper_obs_bounds    = [310.],
                                reward              = ['reward'],
                                episode_length      = 24*3600,
                                random_start_time   = True,
                                warmup_period       = 3600,
                                Ts                  = 900)
    
    def test_stable_baselines_check(self):
        '''Use the environment checker from stable baselines to test 
        the environment. This checks that the environment follows the 
        Gym API. It also optionally checks that the environment is 
        compatible with Stable-Baselines repository.
        
        '''
        
        check_env(self.env, warn=True)
        
    def test_reset_fixed(self):
        '''Test that the environment can reset using a fixed start time
        and a specific warmup period. 
        
        '''
        
        self.env.random_start_time  = False
        self.env.start_time         = 14*24*3600
        self.env.warmup_period      = 3*3600
        
        obs = self.env.reset()
        
        # Check values
        df = pd.DataFrame(data=[obs], index=['obs_reset_fixed'], columns=['value'])
        df.index.name = 'keys'
        ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'reset_fixed.csv')
        self.compare_ref_values_df(df, ref_filepath)
        
    def test_reset_random(self):
        '''Test that the environment can reset using a random start time
        that is out of the specified `excluding_periods`. This test also
        checks that the seed for random initialization works properly. 
        
        '''
        
        self.env.random_start_time  = True
        self.env.warmup_period      = 1*3600
        # Set the excluding periods to be the two first weeks of February
        # and the two first weeks of November
        excluding_periods = [(31*24*3600,  31*24*3600+14*24*3600),
                            (304*24*3600, 304*24*3600+14*24*3600)]
        self.env.excluding_periods = excluding_periods
        random.seed(123456)
        start_times = OrderedDict()
        # Reset hundred times
        for i in range(100):
            obs = self.env.reset()
            start_time = self.env.start_time
            episode = (start_time, start_time+self.env.episode_length)
            for period in excluding_periods:
                # Make sure that the episodes don't overlap with excluding_periods
                assert not(episode[0] < period[1] and period[0] < episode[1]),\
                        'reset is not working properly when generating random times. '\
                        'The episode with starting time {0} and end time {1} '\
                        'overlaps with period {2}. This corresponds to the '\
                        'generated starting time number {3}.'\
                        ''.format(start_time,start_time+self.env.episode_length,period,i)
            start_times[start_time] = obs
            
        # Check values
        df = pd.DataFrame.from_dict(start_times, orient = 'index', columns=['value'])
        df.index.name = 'keys'
        ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'reset_random.csv')
        self.compare_ref_values_df(df, ref_filepath) 
    
    def test_compute_reward_default(self):
        '''Test default method to compute reward.
        
        '''
        rewards = run_baseline.run_reward_default(plot=False)
        
        # Check values
        df = pd.DataFrame(rewards, columns=['value'])
        df.index.name = 'keys'
        ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'rewards_default.csv')
        self.compare_ref_values_df(df, ref_filepath) 
        
    def test_compute_reward_custom(self):
        '''Test custom method to compute reward.
        
        '''
        rewards = run_baseline.run_reward_custom(plot=False)
        
        # Check values
        df = pd.DataFrame(rewards, columns=['value'])
        df.index.name = 'keys'
        ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'rewards_custom.csv')
        self.compare_ref_values_df(df, ref_filepath) 
        
    def test_compute_reward_clipping(self):
        '''Test reward clipping.
        
        '''
        rewards = run_baseline.run_reward_clipping(plot=False)
        
        # Check values
        df = pd.DataFrame(rewards, columns=['value'])
        df.index.name = 'keys'
        ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'rewards_clipping.csv')
        self.compare_ref_values_df(df, ref_filepath) 
        
if __name__ == '__main__':
    utilities.run_tests(os.path.basename(__file__))
