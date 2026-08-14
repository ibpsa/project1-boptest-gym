'''
Module to test features of the OpenAI-Gym interface for BOPTEST.
The BOPTEST bestest_hydronic_heat_pump case needs to be deployed to perform
the tests. 

'''

import unittest
from unittest import mock
import os
import sys
import pandas as pd
import random
import shutil
from testing import utilities
from examples import run_baseline, run_sample, run_save_callback,\
    run_variable_episode, run_vectorized, run_multiaction, train_RL
from collections import OrderedDict
import boptestGymEnv
from boptestGymEnv import BoptestGymEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import A2C, DQN

url = 'http://127.0.0.1'
seed = 123456

class BoptestGymEnvTest(unittest.TestCase, utilities.partialChecks):
    '''Tests the OpenAI-Gym interface for BOPTESTS.
         
    '''
 
    def instantiate_env(self):
        '''Instantiate environment with standard BoptestGymEnv class.
         Used for some tests only.

        '''
        env = BoptestGymEnv(url                 = url,
                            testcase            = 'bestest_hydronic_heat_pump',
                            actions             = ['oveHeaPumY_u'],
                            observations        = {'reaTZon_y':(280.,310.)},
                            reward              = ['reward'],
                            max_episode_length  = 24*3600,
                            random_start_time   = True,
                            warmup_period       = 3600,
                            step_period         = 900)

        return env
    
    def test_summary(self):
        '''
        Test that the environment can print, save, and load a summary 
        describing its most important attributes.  
        
        '''

        # Instantiate environment
        env = self.instantiate_env()

        # Check that we can print the environment summary
        print(env)
        
        # Check that we can save the environment summary
        file_ref = os.path.join(utilities.get_root_path(), 'testing', 'references','summary_ref')
        file_tst = 'summary_tst'
        env.save_summary(file_tst)
        
        # Check that we can load the environment summary. This test only checks sorted keys
        summary = env.load_summary(file_tst)
        for i,k in enumerate(summary.keys()):
            self.compare_ref_json(sorted(dict(summary[k])), file_ref+'_'+str(i)+'.json')
        
        # Remove generated file
        os.remove(file_tst+'.json')

        # stop the environment to not overload the server
        env.stop()

    def test_stable_baselines_check(self):
        '''Use the environment checker from stable baselines to test 
        the environment. This checks that the environment follows the 
        Gym API. It also optionally checks that the environment is 
        compatible with Stable-Baselines3 repository.
        
        '''

        # Instantiate environment
        env = self.instantiate_env()

        check_env(env, warn=True)

        # stop the environment to not overload the server
        env.stop()
   
    def test_reset_fixed(self):
        '''Test that the environment can reset using a fixed start time
        and a specific warmup period. 
        
        '''

        # Instantiate environment
        env = self.instantiate_env()

        env.random_start_time  = False
        env.start_time         = 14*24*3600
        env.warmup_period      = 3*3600
        
        obs, _ = env.reset()
        
        # Check values
        df = pd.DataFrame(data=[obs], index=['obs_reset_fixed'], columns=['value'])
        df.index.name = 'keys'
        ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'reset_fixed.csv')
        self.compare_ref_values_df(df, ref_filepath)

        # stop the environment to not overload the server
        env.stop()

    def test_reset_random(self):
        '''Test that the environment can reset using a random start time
        that is out of the specified `excluding_periods`. This test also
        checks that the seed for random initialization works properly. 
        
        '''

        # Instantiate environment
        env = self.instantiate_env()

        env.random_start_time  = True
        env.warmup_period      = 1*3600
        # Set the excluding periods to be the two first weeks of February
        # and the two first weeks of November
        excluding_periods = [(31*24*3600,  31*24*3600+14*24*3600),
                            (304*24*3600, 304*24*3600+14*24*3600)]
        env.excluding_periods = excluding_periods
        random.seed(seed)
        start_times = OrderedDict()
        # Reset hundred times
        for i in range(100):
            obs, _ = env.reset()
            start_time = env.start_time
            episode = (start_time, start_time+env.max_episode_length)
            for period in excluding_periods:
                # Make sure that the episodes don't overlap with excluding_periods
                assert not(episode[0] < period[1] and period[0] < episode[1]),\
                        'reset is not working properly when generating random times. '\
                        'The episode with starting time {0} and end time {1} '\
                        'overlaps with period {2}. This corresponds to the '\
                        'generated starting time number {3}.'\
                        ''.format(start_time,start_time+env.max_episode_length,period,i)
            start_times[start_time] = obs
            
        # Check values
        df = pd.DataFrame.from_dict(start_times, orient = 'index', columns=['value'])
        df.index.name = 'keys'
        ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'reset_random.csv')
        self.compare_ref_values_df(df, ref_filepath)

        # stop the environment to not overload the server
        env.stop()

    def test_get_reward_default(self):
        '''Test default method to compute reward.
        
        '''

        obs, _, rew = run_baseline.run_reward_default(plot=False)
        self.check_obs_act_rew_kpi(obs=obs,act=None,rew=rew,kpi=None,label='default')

    def test_get_reward_custom(self):
        '''Test custom method to compute reward.
        
        '''

        obs, _, rew = run_baseline.run_reward_custom(plot=False)
        self.check_obs_act_rew_kpi(obs=obs,act=None,rew=rew,kpi=None,label='custom')
        
    def test_get_reward_clipping(self):
        '''Test reward clipping.
        
        '''

        obs, _, rew = run_baseline.run_reward_clipping(plot=False)
        self.check_obs_act_rew_kpi(obs=obs,act=None,rew=rew,kpi=None,label='clipping')

    def test_normalized_observation_wrapper(self):
        '''Test wrapper that normalizes observations.
        
        '''

        obs, _, rew = run_baseline.run_normalized_observation_wrapper(plot=False)
        self.check_obs_act_rew_kpi(obs=obs,act=None,rew=rew,kpi=None,label='normalizedObservationWrapper')

    def test_normalized_action_wrapper(self):
        '''Test wrapper that normalizes actions.
        
        '''

        obs, act, rew = run_sample.run_normalized_action_wrapper(plot=False)
        self.check_obs_act_rew_kpi(obs=obs,act=act,rew=rew,kpi=None,label='normalizedActionWrapper')

    def test_set_scenario(self):
        '''Test that environment can set BOPTEST case scenario.
        
        '''

        obs, _, rew = run_baseline.run_highly_dynamic_price(plot=False)
        self.check_obs_act_rew_kpi(obs=obs,act=None,rew=rew,kpi=None,label='setScenario')
    
    def partial_test_RL(self, algorithm='A2C', mode='load', episode_length_test=1*24*3600,
                        warmup_period_test=1*24*3600, case='simple', training_timesteps=1e5,
                        expert_traj=None, render=False, plot=False):
        '''Test for an RL agent from stable baselines.
        
        Parameters
        ----------
        mode : string, default='load'
            Mode to obtain the RL agent. If `mode=train` then the agent
            will be trained and thus this test case will take long. Setting 
            `mode=load` reduces the testing time considerably by directly 
            loading a pre-trained agent. Independently of whether the 
            agent is trained or not during testing, the results should be 
            exactly the same as far as the seed in `examples.train_RL` 
            is not modified. 
        episode_length_test : integer, default=1*24*3600
            Length of the testing episode. We keep it short for testing,
            only one day is used by default. 
        warmup_period_test : integer, default=1*24*3600
            Length of the initialization period for the test. We keep it 
            short for testing. Only one day is used by default. 
        case : string, default='simple'
            Case to be tested. 
        training_timesteps : int, default=1e5
            Number of timesteps to be used for learning in the test. 
        expert_traj : string, default=None
            Path to expert trajectory if pretraining through behavior 
            cloning is to be used.  
        plot : boolean
            If True the test will plot the time series trajectory. 
        
        '''        
        
        env, model, start_time_tests, _ = train_RL.train_RL(algorithm=algorithm,
                                                            mode=mode, 
                                                            case=case,
                                                            render=render,
                                                            training_timesteps=training_timesteps,
                                                            expert_traj=expert_traj)
        
        obs, act, rew, kpi = \
            train_RL.test_peak(env, model, start_time_tests, 
                               episode_length_test, warmup_period_test, plot=plot)
        if expert_traj is None:
            label = '{0}_{1}_peak'.format(algorithm,case)
        else:
            label = '{0}_{1}_peak_pretrained'.format(algorithm,case)
        self.check_obs_act_rew_kpi(obs,act,rew,kpi,label)
        
        obs, act, rew, kpi = \
            train_RL.test_typi(env, model, start_time_tests, 
                               episode_length_test, warmup_period_test, plot=plot)
        if expert_traj is None:
            label = '{0}_{1}_typi'.format(algorithm,case)
        else:
            label = '{0}_{1}_typi_pretrained'.format(algorithm,case)
        self.check_obs_act_rew_kpi(obs,act,rew,kpi,label)

        # stop the environment to not overload the server
        env.stop()

    def test_A2C_simple(self):
        '''Test simple agent with only one measurement as observation and
        one action. The agent has been trained with 1e5 steps. 
        
        '''
        
        # Use two days in this simple test. All others use only one. 
        self.partial_test_RL(case='simple', algorithm='A2C', episode_length_test=2*24*3600)
        
    def test_A2C_A(self):
        '''Test case A which extends simple case with `time` as observation
        and sets the highly_dynamic price scenario. Hence, this test
        checks the inclusion of time and boundary condition data in the 
        state space. 
        
        '''
        self.partial_test_RL(case='A', algorithm='A2C')
        
    def test_A2C_B(self):
        '''Test case B which extends case A with boundary condition data 
        in the state. Specifically it includes the comfort bounds in the 
        state. Hence, this test checks the inclusion of boundary condition
        data in the state space. 
        
        '''
        self.partial_test_RL(case='B', algorithm='A2C')
        
    def test_A2C_C(self):
        '''Test case C which extends case B with boundary forecast. 
        Specifically it uses a 3 hours forecasting period. Hence, this 
        test checks the use of predictive states. 
        
        '''
        self.partial_test_RL(case='C', algorithm='A2C')
        
    def notest_DQN_D(self):
        '''Test case D which is far more complex than previous cases. 
        Particularly it also uses regressive states discrete action space.  
        
        '''
        self.partial_test_RL(case='D', algorithm='DQN')
        
    def notest_behavior_cloning_cont(self):
        '''Check that an agent using continuous action space (in this case
        we use A2C) can be pretrained using behavior cloning from an 
        expert trajectory that needs to be generated beforehand. The test
        pretrains the agent with 1000 epochs and directly tests its 
        performance without further learning. 
        
        '''
        expert_traj = os.path.join(utilities.get_root_path(),'examples',
                                   'trajectories','expert_traj_cont_28.npz')
        self.partial_test_RL(case='D', algorithm='A2C', mode='train', 
                             training_timesteps=0,
                             expert_traj=expert_traj)
        
    def notest_behavior_cloning_disc(self):
        '''Check that an agent using discrete action space (in this case
        we use DQN) can be pretrained using behavior cloning from an 
        expert trajectory that needs to be generated beforehand. The test
        pretrains the agent with 1000 epochs and directly tests its 
        performance without further learning. 
        
        '''
        expert_traj = os.path.join(utilities.get_root_path(),'examples',
                           'trajectories','expert_traj_disc_28.npz')
        self.partial_test_RL(case='D', algorithm='DQN', mode='train',
                             training_timesteps=0, 
                             expert_traj=expert_traj)

    def test_save_callback(self):
        '''
        Test that the model performance can be monitored and results can be 
        checked and saved as the model improves. This test trains an agent
        for a short period of time, without loading a pre-trained model. 
        Therefore, this test also checks that a RL from stable-baselines3
        can be trained.
        
        '''
        # Define logging directory. Monitoring data and agent model will be stored here
        log_dir = os.path.join(utilities.get_root_path(), 'examples', 'agents', 
                               'monitored_A2C')
        
        # Perform a short training example with callback
        env, _, _ = run_save_callback.train_A2C_with_callback(log_dir=log_dir,
                                                              tensorboard_log=None)  
        
        # Load the trained agent
        model = A2C.load(os.path.join(log_dir, 'best_model'))
        
        # Test one step with the trained model
        obs, _ = env.reset()
        df = pd.DataFrame([model.predict(obs)[0][0]], columns=['value'])
        df.index.name = 'keys'
        ref_filepath    = os.path.join(utilities.get_root_path(), 
                            'testing', 'references', 'save_callback.csv')
        self.compare_ref_values_df(df, ref_filepath)
        
        # Remove model to prove further testing
        shutil.rmtree(log_dir, ignore_errors=True)

        # stop the environment to not overload the server
        env.stop()

    def test_variable_episode(self):
        '''
        Test that a model can be trained using variable episode length. 
        The method that is used to determine whether the episode is 
        truncated or not. This test trains an agent
        for a short period of time, without loading a pre-trained model. 
        Therefore, this test also checks that a RL from stable-baselines3
        can be trained. This test also uses the save callback to check that
        the variable episode length is being effectively used. 
        Notice that this test also checks that child classes can be nested
        since the example redefines the `get_reward` and the 
        `compute_truncated` methods. 
        
        '''
        # Define logging directory. Monitoring data and agent model will be stored here
        log_dir = os.path.join(utilities.get_root_path(), 'examples', 'agents', 
                               'variable_episode_A2C')
        
        # Perform a short training example with callback
        env, _, _ = run_variable_episode.train_A2C_with_variable_episode(log_dir=log_dir,
                                                                         tensorboard_log=None)  
        
        # Load the trained agent
        model = A2C.load(os.path.join(log_dir, 'best_model'))
        
        # Test one step with the trained model
        obs, _ = env.reset()
        df = pd.DataFrame([model.predict(obs)[0][0]], columns=['value'])
        df.index.name = 'keys'
        ref_filepath    = os.path.join(utilities.get_root_path(), 
                            'testing', 'references', 'variable_episode_step.csv')
        self.compare_ref_values_df(df, ref_filepath)
        
        # Check variable lengths
        monitor = pd.read_csv(os.path.join(log_dir,'monitor.csv'),index_col=None)
        monitor = monitor.iloc[1:]
        monitor.reset_index(inplace=True)
        monitor.columns =['reward','episode_length','time']
        
        # Time may vary from one computer to another
        monitor.drop(labels='time',axis=1,inplace=True)
        
        # Utilities require index to have time as index name (even this is not the case here)
        monitor.index.name = 'time'
        
        # Transform to numeric
        monitor = monitor.apply(lambda col:pd.to_numeric(col, errors='coerce'))
        
        # Check that we obtain always same monitoring parameters
        ref_filepath = os.path.join(utilities.get_root_path(), 
                    'testing', 'references', 'variable_episode_monitoring.csv')
        self.compare_ref_timeseries_df(monitor, ref_filepath)
        
        # Remove model to prove further testing
        shutil.rmtree(log_dir, ignore_errors=True)

        # stop the environment to not overload the server
        env.stop()
        
    def check_obs_act_rew_kpi(self, obs=None, act=None, rew=None, kpi=None,
                              label='default'):
        '''Auxiliary method to check for observations, actions, rewards, 
        and/or kpis of a particular test case run. 
        
        '''
        
        # Check observation values
        if obs is not None:
            df = pd.DataFrame(obs)
            df.index.name = 'time' # utilities package requires 'time' as name for index
            ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'observations_{}.csv'.format(label))
            self.compare_ref_timeseries_df(df, ref_filepath) 
        
        # Check actions values
        if act is not None:
            df = pd.DataFrame(act)
            df.index.name = 'time' # utilities package requires 'time' as name for index
            ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'actions_{}.csv'.format(label))
            self.compare_ref_timeseries_df(df, ref_filepath) 
        
        # Check reward values
        if rew is not None:
            df = pd.DataFrame(rew)
            df.index.name = 'time' # utilities package requires 'time' as name for index
            ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'rewards_{}.csv'.format(label))
            self.compare_ref_timeseries_df(df, ref_filepath) 
            
        if kpi is not None:
            df = pd.DataFrame(data=[kpi]).T
            df.columns = ['value']
            df.index.name = 'keys'
            # Time ratio is not checked since depends on the machine where tests are run
            df.drop('time_rat', inplace=True)
            # Drop rows with non-calculated KPIs
            df.dropna(inplace=True)
            ref_filepath = os.path.join(utilities.get_root_path(), 'testing', 'references', 'kpis_{}.csv'.format(label))
            self.compare_ref_values_df(df, ref_filepath)


class BoptestGymVecTest(unittest.TestCase, utilities.partialChecks):
    '''Tests learning with a vectorized environment of BOPTEST-Gym.
         
    '''

    def test_vectorized(self, n_envs=2):
        '''
        Instantiates a vectorized environment with two BOPTEST-Gym environment replicas
        and learns from them when running in parallel using DQN for 100 timesteps.
        It assumes that boptest service is running with at least two workers:
        `docker compose up -d --scale worker=2 web worker provision`.
        Note that this test is also using the `EvalCallback` class from 
        `stable_baselines3.common.callbacks` instead of the
        `boptestGymEnv.SaveAndTestCallback` that we typically use because 
        the former was more convenient for use with vectorized environments. 

        '''

        # Define logging directory. Monitoring data and agent model will be stored here
        log_dir = os.path.join(utilities.get_root_path(), 'examples', 'agents', 'DQN_vectorized')

        # Create BOPTEST-Gym environment replicas, each with its own random seed
        envs = [run_vectorized.make_env(seed=seed+idx) for idx in range(n_envs)]
        
        # Create a vectorized environment using SubprocVecEnv
        venv = SubprocVecEnv(envs)
        
        # Perform a short training example with parallel learning
        run_vectorized.train_DQN_vectorized(venv, log_dir=log_dir)  
        
        # Load the trained agent
        model = DQN.load(os.path.join(log_dir, 'best_model'))
        
        # Test one step with the trained model
        obs = venv.reset()[0]
        df = pd.DataFrame([model.predict(obs)[0]], columns=['value'])
        df.index.name = 'keys'
        ref_filepath    = os.path.join(utilities.get_root_path(), 
                            'testing', 'references', 'vectorized_training.csv')
        self.compare_ref_values_df(df, ref_filepath)
        
        # Remove model to prove further testing
        shutil.rmtree(log_dir, ignore_errors=True)     

class BoptestGymTutorialTest(unittest.TestCase, utilities.partialChecks):
    '''Tests the BOPTEST-Gym tutorial.
         
    '''
    
    def test_tutorial(self):
        '''
        Test the tutorial in the `docs`. The tutorial is written as 
        an ipython notebook so the `nbconvert` package is used to convert 
        the notebook to plain python to execute the test by comparing the
        outputs of some of the notebook cells with references.
        The last section of the tutorial (Gearing Up) is using the DQN 
        algorithm from stable-baselines3 and is used as such in the 
        Quick Start example in the README.md of this repository. 
        Therefore, this is also testing the Quick Start example. 

        '''
        
        from nbconvert.preprocessors import ExecutePreprocessor
        import nbformat

        # Get root directory
        root_dir = utilities.get_root_path()

        # Change working dir to tutorial directory
        run_path = os.chdir(os.path.join(root_dir, 'docs', 'tutorials', 'CCAI_Summer_School_2022'))

        # Path to the notebook file
        notebook_path = os.path.join(root_dir, 'docs', 'tutorials', 'CCAI_Summer_School_2022', 
                                     'Building_Control_with_RL_using_BOPTEST.ipynb')

        # Read the notebook file
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()

        # Execute the notebook cells
        executor = ExecutePreprocessor(timeout=-1, resources={'metadata': {'path': run_path}})
        executed_notebook, _ = executor.preprocess(nbformat.reads(notebook_content, as_version=4),
                                                   resources={'metadata': {'path': run_path}})

        # Test output when requesting test case name
        out_get_name = executed_notebook.cells[41].outputs[0]['text'] 
        self.check_from_cell_output(out_get_name, 'get_name')

        # Check KPIs when testing our Q-algorithm
        out_kpis_Q_alg = executed_notebook.cells[119].outputs[0]['data']['text/plain'] 
        self.check_from_cell_output(out_kpis_Q_alg, 'kpis_Q_alg')

        # Check KPIs when testing DQN algorithm from stable-baselines3
        out_kpis_DQN_alg = executed_notebook.cells[125].outputs[1]['data']['text/plain']
        self.check_from_cell_output(out_kpis_DQN_alg, 'kpis_DQN_alg')

    def check_from_cell_output(self, cell_output, str_output):
        '''Compares a cell output to a reference file. 
        Parameters
        ----------
        cell_output: str
            Content of the cell output that is 
            reformatted in this method to become json
        str_ouput: str
            Tag to identify the reference file of the output

        '''

        import json

        # Conform to the json syntax rules to transform to json
        out = cell_output.replace("\n","").replace("'","\"").replace("None","null")
        
        # Convert string to json
        out_json = json.loads(out)

        # Drop time ratio if it is in output
        if 'time_rat' in out_json:
            del out_json['time_rat']

        # Assign files
        file_ref = os.path.join(utilities.get_root_path(), 'testing', 'references',
                                'tutorial_output_{}.json'.format(str_output))
        
        # Check results
        self.compare_ref_json(out_json, file_ref)

class BoptestGymEnvMultiActTest(unittest.TestCase, utilities.partialChecks):
    ''' Test multi-action training with the `singlezone_commercial_hydronic`
    test case. 
    '''


    def test_training_multi_action(self):
        '''Checks an estimated action after an agent is trained in a multi-action environment.'''

        # Train an agent in a multi-action environment.
        env, model = run_multiaction.train_multiaction()

        # Test one step with the trained model
        obs = env.reset()[0]
        df = pd.DataFrame([model.predict(obs)[0]], columns=['value'])
        df.index.name = 'keys'
        ref_filepath    = os.path.join(utilities.get_root_path(), 
                            'testing', 'references', 'multiaction_training.csv')
        self.compare_ref_values_df(df, ref_filepath)

        # stop the environment to not overload the server
        env.stop()


class StubRequests(object):
    '''Stands in for the requests module.

    Records every call and replays a canned response for the first url that
    contains a registered pattern, so that BOPTEST versions that do and do not
    support an option can both be exercised without a deployed test case.

    '''

    class Response(object):
        def __init__(self, body, status_code=200, text=''):
            self._body, self.status_code, self.text = body, status_code, text

        def json(self):
            if self._body is None:
                raise ValueError('not json')
            return self._body

    def __init__(self):
        self.calls = []
        self.responses = {}

    def _record(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        for pattern, response in self.responses.items():
            if pattern in url:
                return response(kwargs) if callable(response) else response
        return self.Response({'status': 200, 'message': '', 'payload': {}})

    def get(self, url, **kwargs):
        return self._record('get', url, **kwargs)

    def put(self, url, **kwargs):
        return self._record('put', url, **kwargs)

    def post(self, url, **kwargs):
        return self._record('post', url, **kwargs)


class BoptestClientTest(unittest.TestCase):
    '''Tests the client that carries every request to BOPTEST.

    These need no deployed test case: boptestGymEnv's requests module is
    patched with StubRequests for the duration of each test.

    '''

    def setUp(self):
        self.stub = StubRequests()
        patcher = mock.patch.object(boptestGymEnv, 'requests', self.stub)
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- tests ---------------------------------------------------------------

    def _client(self, select_options=None):
        self.stub.responses['select'] = StubRequests.Response({'testid': 'a-test-id'})
        return boptestGymEnv.BoptestClient(url, 'bestest_hydronic_heat_pump',
                                           select_options or {})

    def test_select_options_are_sent_as_given(self):
        '''Test that the select body carries exactly the options asked for, so
        that BOPTEST keeps its own defaults for the rest.

        '''

        self._client()
        self.assertEqual(self.stub.calls[-1][2]['json'], {})
        self.stub.calls = []
        self._client({'direct_step': True, 'fmu_log_level': 0})
        self.assertEqual(self.stub.calls[-1][2]['json'],
                         {'direct_step': True, 'fmu_log_level': 0})

    def test_kpi_subset_when_the_server_supports_it(self):
        '''Test that only the named KPIs are requested and returned.

        '''

        client = self._client()
        self.stub.responses['kpi'] = StubRequests.Response(
            {'payload': {'cost_tot': 1.0, 'tdis_tot': 2.0}})
        kpis = client.kpis(names=boptestGymEnv.REWARD_KPIS)
        self.assertEqual(sorted(kpis), sorted(boptestGymEnv.REWARD_KPIS))
        self.assertEqual(self.stub.calls[-1][2]['params'],
                         {'names': ','.join(boptestGymEnv.REWARD_KPIS)})
        self.assertTrue(client._kpi_subset_supported)

    def test_falls_back_when_the_server_ignores_names(self):
        '''Test the case of a BOPTEST that predates the names parameter: it
        ignores it and returns every KPI, which still contains what the reward
        reads, so the request must succeed rather than fail.

        '''

        client = self._client()
        every_kpi = {'cost_tot': 1.0, 'tdis_tot': 2.0, 'ener_tot': 3.0,
                     'pele_tot': 4.0}
        self.stub.responses['kpi'] = StubRequests.Response({'payload': dict(every_kpi)})
        kpis = client.kpis(names=boptestGymEnv.REWARD_KPIS)
        self.assertEqual(sorted(kpis), sorted(every_kpi))

    def test_falls_back_when_the_server_rejects_names(self):
        '''Test the case of a BOPTEST that rejects the parameter: the client
        asks again for everything and then stops asking for a subset.

        '''

        client = self._client()
        every_kpi = {'cost_tot': 1.0, 'tdis_tot': 2.0, 'ener_tot': 3.0}

        def kpi_response(kwargs):
            if kwargs.get('params'):
                return StubRequests.Response({'status': 400, 'message': 'bad names',
                                       'payload': None})
            return StubRequests.Response({'payload': dict(every_kpi)})

        self.stub.responses['kpi'] = kpi_response
        kpis = client.kpis(names=boptestGymEnv.REWARD_KPIS)
        self.assertEqual(sorted(kpis), sorted(every_kpi))
        self.assertFalse(client._kpi_subset_supported)

        # ...and the next reward does not pay for the failed attempt again
        self.stub.calls = []
        client.kpis(names=boptestGymEnv.REWARD_KPIS)
        self.assertEqual(len(self.stub.calls), 1)
        self.assertIsNone(self.stub.calls[0][2].get('params'))


class LocalClientTest(unittest.TestCase):
    '''Tests the in-process BOPTEST backend.

    Constructing one for real needs pyfmi and a test case FMU, neither of
    which is available here, so these cover what can be checked without them:
    that it refuses clearly when it cannot work, and that it dispatches the
    endpoints the environment uses to the right TestCase methods.

    '''

    class _StubTestCase(object):
        '''Answers with the (status, message, payload) contract of TestCase.'''

        def __init__(self):
            self.calls = []

        def _record(self, name, *args):
            self.calls.append((name,) + args)
            return 200, 'ok', {name: args}

        def get_name(self):             return self._record('get_name')
        def get_measurements(self):     return self._record('get_measurements')
        def get_inputs(self):           return self._record('get_inputs')
        def get_forecast_points(self):  return self._record('get_forecast_points')
        def get_step(self):             return self._record('get_step')
        def get_scenario(self):         return self._record('get_scenario')
        def get_version(self):          return self._record('get_version')
        def set_step(self, step):       return self._record('set_step', step)
        def set_scenario(self, s):      return self._record('set_scenario', s)
        def advance(self, u):           return self._record('advance', u)
        def initialize(self, s, w, warmup_interval=None):
            return self._record('initialize', s, w, warmup_interval)
        def get_results(self, p, s, f): return self._record('get_results', p, s, f)
        def get_forecast(self, p, h, i):return self._record('get_forecast', p, h, i)

        def get_kpis(self, names=None):
            self.calls.append(('get_kpis', names))
            return 200, 'ok', {'cost_tot': 1.0, 'tdis_tot': 2.0}

    def _client(self):
        '''A LocalClient wired to a stub, bypassing __init__ so that no FMU is
        needed.

        '''

        client = boptestGymEnv.LocalClient.__new__(boptestGymEnv.LocalClient)
        client.url = 'local'
        client.testid = 'stub'
        client.case = self._StubTestCase()
        client._workdir = None
        client._kpi_subset_supported = True
        client._warmup_interval_supported = True
        return client

    def test_requires_a_boptest_source_tree(self):
        '''Test that it says what is missing rather than failing on an import.

        '''

        previous = os.environ.pop('BOPTEST_ROOT', None)
        try:
            with self.assertRaises(ValueError) as caught:
                boptestGymEnv.LocalClient(url, 'bestest_hydronic_heat_pump', {})
            self.assertIn('BOPTEST_ROOT', str(caught.exception))
        finally:
            if previous is not None:
                os.environ['BOPTEST_ROOT'] = previous

    def test_dispatches_the_endpoints_the_environment_uses(self):
        '''Test that every endpoint BoptestGymEnv asks for reaches the right
        TestCase method with the right arguments.

        '''

        client = self._client()
        for endpoint, method in [('name', 'get_name'),
                                 ('measurements', 'get_measurements'),
                                 ('inputs', 'get_inputs'),
                                 ('forecast_points', 'get_forecast_points'),
                                 ('step', 'get_step'),
                                 ('scenario', 'get_scenario')]:
            client.get(endpoint)
            self.assertEqual(client.case.calls[-1][0], method)

        client.put('initialize', json={'start_time': 900, 'warmup_period': 60})
        self.assertEqual(client.case.calls[-1], ('initialize', 900, 60, None))
        client.put('step', json={'step': 900})
        self.assertEqual(client.case.calls[-1], ('set_step', 900))
        client.post('advance', json={'oveHeaPumY_u': 0.5})
        self.assertEqual(client.case.calls[-1], ('advance', {'oveHeaPumY_u': 0.5}))
        client.put('results', json={'point_names': ['a'], 'start_time': 0,
                                    'final_time': 900})
        self.assertEqual(client.case.calls[-1], ('get_results', ['a'], 0, 900))
        client.put('forecast', json={'point_names': ['a'], 'horizon': 900,
                                     'interval': 300})
        self.assertEqual(client.case.calls[-1], ('get_forecast', ['a'], 900, 300))

    def test_scenario_keys_are_filled_in(self):
        '''Test that a partial scenario is completed with None, which
        TestCase.set_scenario reads as "leave unchanged" and which it requires
        to be present.

        '''

        client = self._client()
        client.put('scenario', json={'electricity_price': 'constant'})
        sent = client.case.calls[-1][1]
        self.assertEqual(sent['electricity_price'], 'constant')
        for key in ['time_period', 'temperature_uncertainty',
                    'solar_uncertainty', 'seed']:
            self.assertIsNone(sent[key], '{0} was not filled in'.format(key))

    def test_kpi_subset_is_passed_through(self):
        '''Test that requesting named KPIs reaches TestCase, and that asking
        for all of them passes no names.

        '''

        client = self._client()
        client.kpis(names=boptestGymEnv.REWARD_KPIS)
        self.assertEqual(client.case.calls[-1],
                         ('get_kpis', boptestGymEnv.REWARD_KPIS))
        client.kpis()
        self.assertEqual(client.case.calls[-1], ('get_kpis', None))

    def test_warmup_interval_is_passed_through(self):
        '''Test that a warmup grid asked for by the environment reaches
        TestCase.initialize in process, and that it is left out when the
        environment did not ask for one or the local BOPTEST is too old to
        accept it.

        '''

        client = self._client()
        client.put('initialize', json={'start_time': 900, 'warmup_period': 60,
                                       'warmup_interval': 900})
        self.assertEqual(client.case.calls[-1], ('initialize', 900, 60, 900))
        client._warmup_interval_supported = False
        client.put('initialize', json={'start_time': 900, 'warmup_period': 60,
                                       'warmup_interval': 900})
        self.assertEqual(client.case.calls[-1], ('initialize', 900, 60, None))

    def test_unsupported_endpoint_is_reported(self):
        '''Test that an endpoint with no local equivalent says so, rather than
        failing somewhere less obvious.

        '''

        client = self._client()
        for method in [client.get, client.put, client.post]:
            with self.assertRaises(ValueError) as caught:
                method('nonsense')
            self.assertIn('nonsense', str(caught.exception))

    def test_error_from_testcase_becomes_a_runtime_error(self):
        '''Test that a non-200 status from TestCase is raised with its
        message, matching what the REST client does with the same response.

        '''

        client = self._client()
        client.case.get_name = lambda: (400, 'something was wrong', None)
        with self.assertRaises(RuntimeError) as caught:
            client.get('name')
        self.assertIn('something was wrong', str(caught.exception))


if __name__ == '__main__':
    # utilities.run_tests(os.path.basename(__file__))

    test_instance = BoptestGymVecTest()
    test_instance.setUp()
    test_instance.test_vectorized()
