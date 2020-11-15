'''
Created on Jun 4, 2020

@author: Javier Arroyo

'''

import random
import gym
import requests
import numpy as np
import pandas as pd

from gym import spaces
from stable_baselines.common.env_checker import check_env


class BoptestGymEnv(gym.Env):
    '''
    BOPTEST Environment that follows gym interface.
    This environment allows the interaction of RL agents with building
    emulator models from BOPTEST. 
     
    '''
    
    metadata = {'render.modes': ['console']}

    def __init__(self, 
                 url                = 'http://127.0.0.1:5000',
                 actions            = ['oveHeaPumY_u'],
                 observations       = ['reaTZon_y'], 
                 lower_obs_bounds   = [273.],
                 upper_obs_bounds   = [330.],
                 reward             = ['reward'],
                 episode_length     = 3*3600,
                 random_start_time  = False,
                 start_time         = 0,
                 warmup_period      = 0,
                 Ts                 = 900,
                 w                  = 1e8):
        '''
        Parameters
        ----------
        url: string
            Rest API url for communication with the BOPTEST interface
        actions: list
            List of strings indicating the action space. The bounds of 
            each variable from the action space the are retrieved from 
            the overwrite block attributes of the BOPTEST test case
        observations: list
            List of strings indicating the observation space. The observation
            keys must belong to the set of measurements or to the set of 
            forecasting variables of the BOPTEST test case
        lower_obs_bounds: list
            List of floats with the expected lower bounds for the observations.
            It should have the same length as the `observations` argument
        upper_obs_bounds: list
            List of floats with the expected upper bounds for the observations
            It should have the same length as the `observations` argument
        reward: list
            List with string indicating the reward column name in a replay
            buffer of data in case the algorithm is going to use pretraining
        episode_length: integer
            Duration of each episode in seconds
        random_start_time: boolean
            Set to True if desired to use a random start time for each episode. 
        start_time: integer
            Initial fixed episode time in seconds from beginning of the 
            year for each episode. Use in combination with 
            `random_start_time=False` 
        warmup_period: integer
            Desired simulation period to initialize each episode 
        Ts: integer
            Sampling time in seconds
        w: float
            Weight for thermal discomfort in the rewards. This weight is 
            also used to standardize the reward, which is computed as:
            `reward = -(increment_objective_integrand)/self.w` where:
            `objective_integrand = kpis['cost_tot'] + self.w*kpis[tdis_tot]`
            
        '''
        
        super(BoptestGymEnv, self).__init__()
        
        self.url                = url
        self.actions            = actions
        self.observations       = observations
        self.lower_obs_bounds   = lower_obs_bounds
        self.upper_obs_bounds   = upper_obs_bounds
        self.episode_length     = episode_length
        self.random_start_time  = random_start_time
        self.start_time         = start_time
        self.warmup_period      = warmup_period
        self.reward             = reward
        self.Ts                 = Ts
        self.w                  = w
        
        # GET TEST INFORMATION
        # --------------------
        print('\nTEST CASE INFORMATION\n---------------------')
        # Test case name
        self.name = requests.get('{0}/name'.format(url)).json()
        print('Name:\t\t\t\t{0}'.format(self.name))
        # Inputs available
        self.inputs = requests.get('{0}/inputs'.format(url)).json()
        print('Control Inputs:\t\t\t{0}'.format(self.inputs))
        # Measurements available
        self.measurements = requests.get('{0}/measurements'.format(url)).json()
        print('Measurements:\t\t\t{0}'.format(self.measurements))
        # Forecasting variables available
        self.forecasting_vars = list(requests.get('{0}/forecast'.format(url)).json().keys())
        print('Forecasting variables:\t\t\t{0}'.format(self.forecasting_vars))
        # Default simulation step
        self.step_def = requests.get('{0}/step'.format(url)).json()
        print('Default Simulation Step:\t{0}'.format(self.step_def))
        # Default forecast parameters
        self.forecast_def = requests.get('{0}/forecast_parameters'.format(url)).json()
        print('Default Forecast Interval:\t{0} '.format(self.forecast_def['interval']))
        print('Default Forecast Horizon:\t{0} '.format(self.forecast_def['horizon']))
        # --------------------
        
        # Define action space. It must be a gym.space object
        lower_input_bounds = []
        upper_input_bounds = []
        for inp in self.actions:
            assert inp in self.inputs.keys()
            lower_input_bounds.append(self.inputs[inp]['Minimum'])
            upper_input_bounds.append(self.inputs[inp]['Maximum'])
            
        self.action_space = spaces.Box(low  = np.array(lower_input_bounds), 
                                       high = np.array(upper_input_bounds), 
                                       dtype= np.float32)
        
        # Define observation space. It must be a gym.space object
        for obs in self.observations:
            if not (obs in self.measurements.keys() or obs in self.forecasting_vars):
                raise ReferenceError(\
                 '"{0}" does not belong to neither the set of '\
                 'test case measurements nor to the set of '\
                 'forecasted variables. \n'\
                 'Set of measurements: \n{1}\n'\
                 'Set of forecasting variables: \n{2}'.format(obs, 
                                                              list(self.measurements.keys()), 
                                                              self.forecasting_vars))

        self.observation_space = spaces.Box(low  = np.array(self.lower_obs_bounds), 
                                            high = np.array(self.upper_obs_bounds), 
                                            dtype= np.float32)    

    def reset(self, seed=1):
        '''
        Important: the observation must be a numpy array
        
        Parameters
        ----------
        seed: int
            Seed for random start time 
            
        Returns
        -------
        meas: numpy array
            Measurements at the end of initialization
         
        '''        
        
        # Assign random start_time if it is None
        if self.random_start_time:
            random.seed(seed)
            self.start_time = random.randint(0, 3.154e+7-self.episode_length)
        
        # Initialize the building simulation
        res = requests.put('{0}/initialize'.format(self.url), 
                           data={'start_time':self.start_time,
                                 'warmup_period':self.warmup_period}).json()
        
        # Set simulation step
        requests.put('{0}/step'.format(self.url), data={'step':self.Ts})
        
        # Initialize objective integrand
        self.objective_integrand = 0.
        
        # Get measurements at the end of the initialization period
        meas = self.get_measurements(res)
        
        return meas

    def step(self, action):
        '''
        Advance the simulation one time step
        
        Parameters
        ----------
        action: list
            List of actions computed by the agent to be implemented 
            in this step
            
        Returns
        -------
        meas: numpy array
            Measurements at the end of this time step
        reward: float
            Reward for the state-action pair implemented
        done: boolean
            True if episode is finished after this step
        info: dictionary
            Additional information for this step
        
        '''
        
        # Initialize inputs to send through BOPTEST Rest API
        u = {}
        
        # Assign values to inputs if any
        for i, act in enumerate(self.actions):
            # Assign value
            u[act] = action[i]
            
            # Indicate that the input is active
            u[act.replace('_u','_activate')] = 1.
                
        # Advance a BOPTEST simulation
        res = requests.post('{0}/advance'.format(self.url), data=u).json()
        
        # Define whether we've finished the episode
        done = res['time'] >= self.start_time + self.episode_length
        
        # Compute reward of this (state-action-state') tuple
        reward = self.compute_reward()
                
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        # Get measurements at the end of this time step
        meas = self.get_measurements(res)
                
        return meas, reward, done, info
    
    def render(self, mode='console'):
        '''
        Renders the process evolution 
        
        Parameters
        ----------
        mode: string
            Mode to be used for the renderization
        
        '''
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass
    
    def compute_reward(self):
        '''
        Compute the reward of last state-action-state' tuple. The 
        reward is implemented as the negated increase in the objective
        integrand function. In turn, this objective integrand function 
        is calculated as the sum of the total operational cost plus
        the weighted discomfort. The value to weight discomfort
        can be accessed through `self.w` and is used to compensate 
        for the different orders of magnitude of cost and discomfort. 
        This weight is also used to rescale the rewards. Notice that
        the rewards are rescaled without shifting mean as that would
        affect agent's will to live. 
        
        Returns
        -------
        Reward: float
            Reward of last state-action-state' tuple
        
        Notes
        -----
        This method should be changed to be an abstract method so that
        users can redefine it as desired. 
        
        '''
        
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi'.format(self.url)).json()
        
        # Calculate objective integrand function at this point
        objective_integrand = kpis['cost_tot'] + self.w*kpis['tdis_tot']
        
        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)/self.w
        
        self.objective_integrand = objective_integrand
        
        return reward
        
    def get_measurements(self, res):
        '''
        Get measurement outputs in the right format and assign the 
        simulation keys to them. Add noise if any. Concatenate the 
        obtained mesurements. 
        
        Parameters
        ----------
        res: dictionary
            
        
        Returns
        -------
        meas: float
            Reformatted observations 
        
        '''
        
        # Get reults at the end of the simulation step
        observations = []
        for obs in self.observations:
            observations.append(res[obs])
            
        # Reformat observations
        meas = np.array(observations).astype(np.float32)
                
        return meas
    
    def reformat_expert_traj(self, file_path='data.csv'):
        '''
        Reformats expert trajectory from a csv file to the npz format 
        required by Stable Baselines algorithms to be pre-trained.   
        
        Parameters
        ----------
        file_path: string
            path to csv file containing data
            
        Returns
        -------
        numpy_dict: numpy dictionary
            Numpy dictionary with the reformatted data
        
        Notes
        -----
        The resulting reformatted data considers only one episode from
        a long trajectory (a long time series). No recurrent policies 
        supported (mask and state not defined). 
        
        '''
        
        # We consider only one episode of index 0 that is never done
        n_episodes = 1
        ep_idx = 0
        done = False
        
        # Initialize data in the episode
        actions = []
        observations = []
        rewards = []
        episode_returns = np.zeros((n_episodes,))
        episode_starts = []
        
        # Initialize the only episode that we use
        episode_starts.append(True)
        reward_sum = 0.0

        df = pd.read_csv(file_path)
        for row in df.index:
            # Retrieve step information from csv
            obs     = df.loc[row, self.observations]
            action  = df.loc[row, self.actions]
            reward  = df.loc[row, self.reward]
            
            if obs.hasnans or action.hasnans or reward.hasnans:
                raise ValueError('Nans found in row {}'.format(row))
            
            # Append to data
            observations.append(np.array(obs))
            actions.append(np.array(action))
            rewards.append(np.array(reward))
            episode_starts.append(np.array(done))
            
            reward_sum += reward
        
        # This is hard coded as we only support one episode so far but
        # here we could implement some functionality for creating different 
        # episodes from csv data
        done = True
        if done:
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
    
        if isinstance(self.observation_space, spaces.Box):
            observations = np.concatenate(observations).reshape((-1,) + self.observation_space.shape)
        elif isinstance(self.observation_space, spaces.Discrete):
            observations = np.array(observations).reshape((-1, 1))
    
        if isinstance(self.action_space, spaces.Box):
            actions = np.concatenate(actions).reshape((-1,) + self.action_space.shape)
        elif isinstance(self.action_space, spaces.Discrete):
            actions = np.array(actions).reshape((-1, 1))
    
        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts[:-1])
    
        assert len(observations) == len(actions)
    
        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }  # type: Dict[str, np.ndarray]
    
        for key, val in numpy_dict.items():
            print(key, val.shape)
    
        np.savez(file_path.split('.')[-2], **numpy_dict)
        
        return numpy_dict

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    '''This wrapper converts the Box observation space into a Discrete 
    observation space.
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = DiscretizedObservationWrapper(env, n_bins_obs=10)`
    
    '''
    
    def __init__(self, env, n_bins_obs=10):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        n_bins_obs: integer
            Number of bins to be used in the transformed observation 
            space for each observation. 
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
        # Assign attributes (env already assigned)
        self.n_bins_obs = n_bins_obs

        # Assert that original observation space is a Box space
        assert isinstance(env.observation_space, spaces.Box)
        
        # Get observation space bounds
        low     = self.observation_space.low
        high    = self.observation_space.high
        
        # Calculate dimension of observation space
        n_obs = low.flatten().shape[0]
        
        # Obtain values of discretized observation space
        self.val_bins_obs   = [np.linspace(l, h, n_bins_obs + 1) for l, h in
                               zip(low.flatten(), high.flatten())]
        
        # Instantiate discretized observation space
        self.observation_space = spaces.Discrete(n_bins_obs ** n_obs)

    def observation(self, observation):
        '''
        This method accepts a single parameter (the 
        observation to be modified) and returns the modified observation.
        
        Parameters
        ----------
        observation: 
            Observation in the original environment observation space format 
            to be modified.
        
        Returns
        -------
            Modified observation returned by the wrapped environment. 
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ObservationWrapper` parent class is doing in `gym.core`:
        
        '''
        
        # Get the bin indexes for each element of this observation
        indexes = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins_obs)]
        
        # Convert to one number for the wrapped environment
        observation_wrapper = sum([index * ((self.n_bins_obs + 1) ** obs_i) for obs_i, index in enumerate(indexes)])
        
        return observation_wrapper
    
class DiscretizedActionWrapper(gym.ActionWrapper):
    '''This wrapper converts the Box action space into a Discrete action 
    space. 
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = DiscretizedActionWrapper(env, n_bins_act=10)`
    
    '''
    def __init__(self, env, n_bins_act=10):
        '''Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        n_bins_obs: integer
            Number of bins to be used in the transformed observation space
            for each observation. 
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
        # Assign attributes (env already assigned)
        self.n_bins_act = n_bins_act

        # Assert that original action space is a Box space
        assert isinstance(env.action_space, spaces.Box)
        
        # Get observation space bounds
        low     = self.action_space.low
        high    = self.action_space.high
        
        # Calculate dimension of observation space
        n_act = low.flatten().shape[0]
        
        # Obtain values of discretized action space
        self.val_bins_act   = [np.linspace(l, h, n_bins_act + 1) for l, h in
                               zip(low.flatten(), high.flatten())]
        
        # Instantiate discretized action space
        self.action_space = spaces.Discrete(n_bins_act ** n_act)

    def action(self, action_wrapper):
        '''This method accepts a single parameter (the modified action
        in the wrapper format) and returns the action to be passed to the 
        original environment. 
        
        Parameters
        ----------
        action_wrapper: 
            Action in the modified environment action space format 
            to be reformulated back to the original environment format.
        
        Returns
        -------
            Action in the original environment format.  
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ActionWrapper` parent class is doing in `gym.core`:
        
        Implement something here that performs the following mapping:
        DiscretizedObservationWrapper.action_space --> DiscretizedActionWrapper.action_space
        
        '''
        
        # Get the bin indexes for each element of this observation
        indexes = [np.digitize([x], bins)[0]
                  for x, bins in zip(action_wrapper.flatten(), self.val_bins_act)]
        
        # Return values from indexes
        action = []
        for act_i, index in enumerate(indexes):
            action.append(self.val_bins_act[act_i][index-1])
        action = np.asarray(action).astype(self.env.action_space.dtype)
        
        return action
    
if __name__ == "__main__":
    
    # Instantiate the env    
    env = BoptestGymEnv()

    # Check the environment
    check_env(env, warn=True)
    obs = env.reset()
    env.render()
    print('Observation space: {}'.format(env.observation_space))
    print('Action space: {}'.format(env.action_space))
    