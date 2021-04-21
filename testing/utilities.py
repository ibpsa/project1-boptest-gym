# -*- coding: utf-8 -*-
"""
This module contains testing utilities used throughout test scripts, including
common functions and partial classes. This is mainly a copy of the 
utilities module that is also used within the BOPTEST repository. 

"""

import os
import unittest
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

def get_root_path():
    '''Returns the path to the root repository directory.
    
    '''
    
    testing_path = os.path.dirname(os.path.realpath(__file__));
    root_path = os.path.split(testing_path)[0]
    
    return root_path;

def clean_up(dir_path):
    '''Cleans up the .fmu, .mo, .txt, .mat, .json files from directory.

    Parameters
    ----------
    dir_path : str
        Directory path to clean up
        
    '''

    files = os.listdir(dir_path)
    for f in files:
        if f.endswith('.fmu') or f.endswith('.mo') or f.endswith('.txt') or f.endswith('.mat') or f.endswith('.json'):
            os.remove(os.path.join(dir_path, f))
            
def run_tests(test_file_name):
    '''Run tests and save results for specified test file.
    
    Parameters
    ----------
    test_file_name : str
        Test file name (ends in .py)
    
    '''

    # Load tests
    test_loader = unittest.TestLoader()
    suite = test_loader.discover(os.path.join(get_root_path(),'testing'), pattern = test_file_name)
    num_cases = suite.countTestCases()
    # Run tests
    print('\nFound {0} tests to run in {1}.\n\nRunning...'.format(num_cases, test_file_name))
    result = unittest.TextTestRunner(verbosity = 1).run(suite);
    # Parse and save results
    num_failures = len(result.failures)
    num_errors = len(result.errors)
    num_passed = num_cases - num_errors - num_failures
    log_json = {'TestFile':test_file_name, 'NCases':num_cases, 'NPassed':num_passed, 'NErrors':num_errors, 'NFailures':num_failures, 'Failures':{}, 'Errors':{}}
    for i, failure in enumerate(result.failures):
        log_json['Failures'][i]= failure[1]
    for i, error in enumerate(result.errors):
        log_json['Errors'][i]= error[1]
    log_file = os.path.splitext(test_file_name)[0] + '.log'
    with open(os.path.join(get_root_path(),'testing',log_file), 'w') as f:
        json.dump(log_json, f)
        
def compare_references(vars_timeseries = ['reaTRoo_y'],
                       refs_old = 'references_old',
                       refs_new = 'references'):
    '''Method to perform visual inspection on how references have changed
    with respect to a previous version.

    Parameters
    ----------
    vars_timeseries : list
        List with strings indicating the variables to be plotted in time
        series graphs.
    refs_old : str
        Name of the folder containing the old references.
    refs_new : str
        Name of the folder containing the new references.

    '''

    dir_old = os.path.join(get_root_path(), 'testing', refs_old)

    for subdir, _, files in os.walk(dir_old):
        for filename in files:
            f_old = os.path.join(subdir, filename)
            f_new = os.path.join(subdir.replace(refs_old,refs_new), filename)
            if not os.path.exists(f_new):
                print('File: {} has not been compared since it does not exist anymore.'.format(f_new))

            elif not f_old.endswith('.csv'):
                print('File: {} has not been compared since it is not a csv file.'.format(f_old))

            else:
                df_old = pd.read_csv(f_old)
                df_new = pd.read_csv(f_new)

                if not('time' in df_old.columns or 'keys' in df_old.columns):
                    print('File: {} has not been compared because the format is not recognized.'.format(f_old))
                else:
                    if 'time' in df_old.columns:
                        df_old.drop('time', axis=1, inplace=True)
                        df_new.drop('time', axis=1, inplace=True)
                        kind = 'line'
                        vars_to_plot = vars_timeseries
                    elif 'keys' in df_old.columns:
                        df_old = df_old.set_index('keys')
                        df_new = df_new.set_index('keys')
                        kind = 'bar'
                        vars_to_plot = df_old.columns

                    if 'kpis_' in filename:
                        fig, axs = plt.subplots(nrows=1, ncols=len(df_old.index), figsize=(10,8))
                        for i,k in enumerate(df_old.index):
                            axs[i].bar(0, df_old.loc[k,'value'], label='old', alpha=0.5, color='orange')
                            axs[i].bar(0, df_new.loc[k,'value'], label='new', alpha=0.5, color='blue')
                            axs[i].set_title(k)
                        fig.suptitle(str(f_new))
                        plt.legend()
                    else:
                        if any([v in df_old.keys() for v in vars_to_plot]):
                            for v in vars_to_plot:
                                if v in df_old.keys():
                                    _, ax = plt.subplots(1, figsize=(10,8))
                                    df_old[v].plot(ax=ax, label='old '+v, kind=kind, alpha=0.5, color='orange')
                                    df_new[v].plot(ax=ax, label='new '+v, kind=kind, alpha=0.5, color='blue')
                                    ax.set_title(str(f_new))
                                    ax.legend()
                        else:
                            print('File: {} has not been compared because it does not contain any of the variables to plot'.format(f_old))

    plt.show()
                
class partialChecks(object):
    '''This partial class implements common ref data check methods.
    
    '''
    
    def compare_ref_timeseries_df(self, df, ref_filepath):
        '''Compare a timeseries dataframe to a reference csv.
        
        Parameters
        ----------
        df : pandas DataFrame
            Test dataframe with "time" as index.
        ref_filepath : str
            Reference file path relative to testing directory.
            
        Returns
        -------
        None
        
        '''
        
        # Check time is index
        assert(df.index.name == 'time')
        # Perform test
        if os.path.exists(ref_filepath):
            # If reference exists, check it
            df_ref = pd.read_csv(ref_filepath, index_col='time')   
            # Ensure that both, df and df_ref have strings as keys
            df.columns      = [str(c) for c in df.columns.to_list()]
            df_ref.columns  = [str(c) for c in df_ref.columns.to_list()]
            # Check all keys in reference are in test
            for key in df_ref.columns.to_list():
                self.assertTrue(key in df.columns.to_list(), 'Reference key {0} not in test data.'.format(key))
            # Check all keys in test are in reference
            for key in df.columns.to_list():
                self.assertTrue(key in df_ref.columns.to_list(), 'Test key {0} not in reference data.'.format(key))
            # Check trajectories
            for key in df.columns:
                y_test = self.create_test_points(df[key]).to_numpy()
                y_ref = self.create_test_points(df_ref[key]).to_numpy()
                results = self.check_trajectory(y_test, y_ref)
                self.assertTrue(results['Pass'], '{0} Key is {1}.'.format(results['Message'],key))
        else:
            # Otherwise, save as reference
            df.to_csv(ref_filepath)
            
        return None
    
    def compare_ref_json(self, json_test, ref_filepath):
            '''Compare a json to a reference json saved as .json.
            
            Parameters
            ----------
            json_test : Dict
                Test json in the form of a dictionary.
            ref_filepath : str
                Reference .json file path relative to testing directory.
                
            Returns
            -------
            None
            
            '''
            
            # Perform test
            if os.path.exists(ref_filepath):
                # If reference exists, check it
                with open(ref_filepath, 'r') as f:
                    json_ref = json.load(f)               
                self.assertTrue(json_test==json_ref, 'json_test:\n{0}\ndoes not equal\njson_ref:\n{1}'.format(json_test, json_ref))
            else:
                # Otherwise, save as reference
                with open(ref_filepath, 'w') as f:
                    json.dump(json_test,f)
                
            return None
        
    def compare_ref_values_df(self, df, ref_filepath):
        '''Compare a values dataframe to a reference csv.
        
        Parameters
        ----------
        df : pandas DataFrame
            Test dataframe with a number of keys as index paired with values.
        ref_filepath : str
            Reference file path relative to testing directory.
            
        Returns
        -------
        None
        
        '''
        
        # Check keys is index
        assert(df.index.name == 'keys')
        assert(df.columns.to_list() == ['value'])
        # Perform test
        if os.path.exists(ref_filepath):
            # If reference exists, check it
            df_ref = pd.read_csv(ref_filepath, index_col='keys')           
            for key in df.index.values:
                y_test = [df.loc[key,'value']]
                y_ref = [df_ref.loc[key,'value']]
                results = self.check_trajectory(y_test, y_ref)
                self.assertTrue(results['Pass'], '{0} Key is {1}.'.format(results['Message'],key))
        else:
            # Otherwise, save as reference
            df.to_csv(ref_filepath)
            
        return None
    
    def check_trajectory(self, y_test, y_ref):
        '''Check a numeric trajectory against a reference with a tolerance.
        
        Parameters
        ----------
        y_test : list-like of numerics
            Test trajectory
        y_ref : list-like of numerics
            Reference trajectory
            
        Returns
        -------
        result : dict
            Dictionary of result of check.
            {'Pass' : bool, True if ErrorMax <= tol, False otherwise.
             'ErrorMax' : float or None, Maximum error, None if fail length check
             'IndexMax' : int or None, Index of maximum error,None if fail length check
             'Message' : str or None, Message if failed check, None if passed.
            }
        
        '''
    
        # Set tolerance
        tol = 1e-3
        # Initialize return dictionary
        result =  {'Pass' : True,
                   'ErrorMax' : None,
                   'IndexMax' : None,
                   'Message' : None}
        # First, check that trajectories are same length
        if len(y_test) != len(y_ref):
            result['Pass'] = False
            result['Message'] = 'Test and reference trajectory not the same length.'
        else:
            # Initialize error arrays
            err_abs = np.zeros(len(y_ref))
            err_rel = np.zeros(len(y_ref))
            err_fun = np.zeros(len(y_ref))
            # Calculate errors
            for i in range(len(y_ref)):
                # Absolute error
                err_abs[i] = np.absolute(y_test[i] - y_ref[i])
                # Relative error
                if (abs(y_ref[i]) > 10 * tol):
                    err_rel[i] = err_abs[i] / abs(y_ref[i])
                else:
                    err_rel[i] = 0
                # Total error
                err_fun[i] = err_abs[i] + err_rel[i]
                # Assess error
                err_max = max(err_fun);
                i_max = np.argmax(err_fun);
                if err_max > tol:
                    result['Pass'] = False
                    result['ErrorMax'] = err_max,
                    result['IndexMax'] = i_max,
                    result['Message'] = 'Max error ({0}) in trajectory greater than tolerance ({1}) at index {2}. y_test: {3}, y_ref:{4}'.format(err_max, tol, i_max, y_test[i_max], y_ref[i_max])
        
        return result
    
    def create_test_points(self, s,n=500):
        '''Create interpolated points to test of a certain number.
        
        Useful to reduce number of points to test and to avoid failed tests from
        event times being slightly different.
    
        Parameters
        ----------
        s : pandas Series
            Series containing test points to create, with index as time floats.
        n : int, optional
            Number of points to create
            Default is 500
            
        Returns
        -------
        s_test : pandas Series
            Series containing interpolated data    
    
        '''
        
        # Get data
        data = s.to_numpy()
        index = s.index.values
        # Make interpolated index
        t_min = index.min()
        t_max = index.max()
        t = np.linspace(t_min, t_max, n)
        # Interpolate data
        data_interp = np.interp(t,index,data)
        # Use at most 8 significant digits
        data_interp = [ float('{:.8g}'.format(x)) for x in data_interp ]
        # Make Series
        s_test = pd.Series(data=data_interp, index=t)
        
        return s_test
    
    def results_to_df(self, results):
        '''Convert results from boptest into pandas DataFrame timeseries.
        
        Parameters
        ----------
        results: dict
            Dictionary of results provided by boptest api "/results".
        
        Returns
        -------
        df: pandas DataFrame
            Timeseries dataframe object with "time" as index in seconds.
            
        '''
        
        df = pd.DataFrame()
        for s in ['y','u']:
            for x in results[s].keys():
                if x != 'time':
                    df = pd.concat((df,pd.DataFrame(data=results[s][x], index=results['y']['time'],columns=[x])), axis=1)
        df.index.name = 'time'
        
        return df        
