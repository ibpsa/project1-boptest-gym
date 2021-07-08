

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from state_estimator.ukf import UKF, UKFOptions

class Observer_UKF(object):
    '''
    This class returns observations from an unscented Kalman filter
    
    '''
    
    def __init__(self, parent=None,
                 model_ukf = None,
                 cov_meas_noise = 0.3**2,
                 cov_proc_noise = 0.4**2,
                 alpha = 0.01,
                 beta = 2.0,
                 kappa = 0.0,
                 stai=None,
                 pars_json_file=None):
        '''
        Constructor
        
        Parameters
        ----------
        parent: Controller_MPC
            Instance of a Controller class. The observer object will
            inherit all the attributes from the controller.
        model_ukf: fmu
            Model to be used for predictions. 
        cov_meas_noise: float
            Expected covariance of the measurement noise
        cov_proc_noise: float
            Expected covariance of the process noise
        alpha: float
            Scaling parameter for distribution of sigma points, should
            be between 0 and 1. 
        beta: float
            Parameter for including prior knowledge of state variance,
            where beta=2 is optimal for Gaussian distributions. 
        kappa: float
            Secondary scaling parameter, used to ensure semi-positive
            definiteness of covariance matrix. Usually set to zero. 
        stai: Pandas data frame
            data frame of the form 
            `self.stai = pd.DataFrame(index=[self.time_sim[0]], columns=self.stat_names)`
            with the initial states of the controller model. If None,
            a default of 293.15 is used for each state. 
        pars_json_file: String
            Path to file with the parameters obtained externally by system 
            identification which value needs to be set in the model 
        
        Notes
        -----
        The expected covariance of the initial states is calculated as 
        the sum of the expected covariance of the measurement noise
        and the expected covariance of the process noise.  
            
        '''
        
        self.model_ukf = model_ukf
        
        # Inherit attributes from parent class
        for k,v in parent.__dict__.items():
            self.__dict__[k] = v
        
        # Assign expected covariances of measurement and process noise
        self.cov_meas_noise = cov_meas_noise   
        self.cov_proc_noise = cov_proc_noise   
        self.cov_stat_noise = cov_meas_noise + cov_proc_noise   

        # Set options for UKF
        options = UKFOptions()
        P_n = {meas: self.cov_meas_noise for meas in self.meas_names}
        P_v = {stat: self.cov_proc_noise for stat in self.stat_names} 
        P_0 = {stat: self.cov_stat_noise for stat in self.stat_names}
        self.alpha  = alpha
        self.beta   = beta
        self.kappa  = kappa
        
        # Update the options with the 
        options.update(alpha=self.alpha, beta=self.beta, kappa=self.kappa, 
                       P_0=P_0, P_v=P_v, P_n=P_n)        
        
        # Set initial state estimate, measured variables, and sampling interval
        if stai is None:
            self.stai = pd.DataFrame(index=[self.time_sim[0]], columns=self.stat_names)
            for stat in self.stat_names:
                self.stai.loc[self.time_sim[0], stat] = 293.15
        else:
            self.stai = stai
            
        self.conf = pd.DataFrame(index=[self.time_sim[0]], columns=self.stat_names)
        for stat in self.stat_names:
            self.conf.loc[self.time_sim[0], stat] = np.sqrt(self.cov_stat_noise)
            
        stai_dict = {stat: self.stai.loc[self.time_sim[0], stat] for stat in self.stat_names} 
        
        # Load the parameter and set their values into model_sim, model_mpc and model_ukf
        if pars_json_file is not None:
            with open(pars_json_file, 'r') as fopen:
                pars_opt = json.load(fopen)
            self.pars_fixed = {str(k):v for k,v in pars_opt.items()}
            self.model_ukf.set(list(self.pars_fixed.keys()), list(self.pars_fixed.values()))
        
        def derive_start_par(var_name='bui.zon.capZon.heaPor.T'):
            '''
            This method defines the rules to derive the start value 
            parameters from the variable names and can be customized. 
            For this particular case we want to go
            from --> 'bui.zon.capZon.heaPor.T'
            to   --> 'bui.zon.capZon.TSta'
            For each of the model states. 
            
            Parameters
            ----------
            var_name: string
                variable for which the start name parameter is to be 
                derived
                
            '''
            
            sta_par_name = var_name.replace('.heaPor.T','.TSta')
            
            return sta_par_name
        
        # Create an UKF object

        self.ukf = UKF(self.model_ukf, stai_dict, self.meas_names, self.Ts, 
                       options, start_suffix=derive_start_par,
                       pars_fixed=self.pars_fixed)

        # Derive positions of states which are sorted alphabetically in ukf
        self.x_pos = {}
        for i,state in enumerate(self.ukf.x):    
            self.x_pos[state.get_name()] = i
            
        # Derive positions of outputs which are sorted alphabetically in ukf
        self.y_pos = {}
        for i,measurement in enumerate(self.ukf.mes):    
            self.y_pos[measurement.get_name()] = i

        # Initialize simulation variables
        self.outp_sim = pd.DataFrame() # Predicted outputs 
        self.stap_sim = pd.DataFrame() # Predicted states 
        self.stai_sim = pd.DataFrame() # Initial states after update
        self.meas_sim = pd.DataFrame() # Measurements
        self.conf_sim = pd.DataFrame() # Confidence interval
        # self.time_zoh_stp = pd.date_range(start=self.time_sim[0], end=self.time_sim_0+1, freq='min')
                                
    def observe(self, meas_stp, cInp_stp=None, dist_stp=None):
        '''
        Perform prediction and update steps to estimate the actual 
        initial states of the controller model.
        
        Parameters
        ----------
        meas_stp: pandas data frame
            Measurements at current time step of the form:
            `meas_stp = pd.DataFrame(index=[time_now], columns=self.meas_names)`
        cInp_stp: pandas data frame
            Controllable inputs applied since the last time step. 
        dist_stp: pandas data frame
            Disturbances during last step.
        
        Returns
        -------
        stai_stp: pandas data frame
            Estimation of the initial states at the current time. Format:
            `stai_stp = pd.DataFrame(index=[time_now], columns=self.stat_names)`
        
        '''
        
        # Retrieve actual and previous times 
        time_now = meas_stp['time']
        
        # Prediction of the ACTUAL time step
        u=self.create_input_object(cInp_stp, dist_stp)
        self.ukf.predict(u=u) 
        
        m = {}
        for k,v in self.meas_map.items():
            m[k] = meas_stp[v]
        
        # Update of the ACTUAL time step
        stai_stp_dict, conf_stp_dict = self.ukf.update(m)  
        
        if False:
            # Initialize the predicted outputs, predicted states, initial states and confidence intervals
            outp_stp = pd.DataFrame(index=time_now, columns=self.meas_names)
            stap_stp = pd.DataFrame(index=time_now, columns=self.stat_names)
            stai_stp = pd.DataFrame(index=time_now, columns=self.stat_names)
            conf_stp = pd.DataFrame(index=time_now, columns=self.stat_names)
            
            # Convert to pandas format
            for stat in self.stat_names:
                stap_stp.loc[time_now, stat] = self.ukf.xp[self.x_pos[stat]]*\
                                               self.ukf.x[self.x_pos[stat]].get_nominal_value()
                stai_stp.loc[time_now, stat] = stai_stp_dict[stat]
                conf_stp.loc[time_now, stat] = conf_stp_dict[stat]
    
            for meas in self.meas_names:    
                outp_stp.loc[time_now, meas] = self.ukf.yp[self.y_pos[meas]]*\
                                               self.ukf.mes[self.y_pos[meas]].get_nominal_value()
    
            # Store results
            self.outp_sim = pd.concat([self.outp_sim, outp_stp])
            self.stap_sim = pd.concat([self.stap_sim, stap_stp])
            self.meas_sim = pd.concat([self.meas_sim, meas_stp])
            self.stai_sim = pd.concat([self.stai_sim, stai_stp])
            self.conf_sim = pd.concat([self.conf_sim, conf_stp])
                
        return stai_stp_dict
    
    def create_input_object(self, cInp_dict, dist_dict):
        """
        Creates an input object compliant with pyfmi standard for 
        simulation of fmu model
        
        Parameters
        ----------
        cInp_dict : dict    
            data to be used as controllable input for the simulation
        dist_dict : dict    
            data to be used as disturbance input for the simulation

        Returns
        -------
        object: list
            returns the object that is required as argument in an FMU 
            simulate call.        
            
        """
        
        time = cInp_dict['time']
        cInp_dict.pop('time')
        dist_dict.pop('time')
        
        all_inputs = {}
        all_inputs.update(cInp_dict)
        all_inputs.update(dist_dict)
        
        udf = pd.DataFrame(all_inputs)
    
        for u in udf.columns:
            u = np.column_stack((time, udf))            
        
        return [udf.columns, u]
        
        #=============================================================
        # for u in inputs_dict.keys():
        #     u = np.column_stack((time, inputs_dict))
        #     return [inputs_dict.keys(), u]
        #=============================================================
    
    def plot_observations(self, true_values, show_hidden=False):
        '''
        Plot the initial states estimated, the measurements gotten, 
        and the true values. Notice that the true_values is something
        that a state observer does not perceive, so need to be 
        provided externally for this plot.
        
        Parameters
        ----------
        true_values: pandas data frame
            time series of the available true state values
        show_hidden: boolean
            set to True to also show the hidden states
        
        '''
        
        import pandas.tseries.converter as converter
        c = converter.DatetimeConverter()
        
        plt.figure('Observations')
        plt.subplot(2, 1, 1)
        for meas in self.meas_names:
            plt.plot(self.meas_sim.index, self.meas_sim.loc[:,meas], 'rx', label='measured_' +meas)
            plt.plot(self.outp_sim.index, self.outp_sim.loc[:,meas], label='predicted_'+meas, marker='s')
        if show_hidden:
            for stat in self.stat_names:
                plt.plot(self.stap_sim.index, self.stap_sim.loc[:,stat], label='predicted_'+stat, marker='o')
                plt.plot(self.stai_sim.index, self.stai_sim.loc[:,stat], label='updated_'+stat)
                try:
                    plt.fill_between(c.convert(self.stai_sim.index,None,None), 
                                     c.convert(self.stai_sim.loc[:,stat] - self.conf_sim.loc[:,stat],None,None),  
                                     c.convert(self.stai_sim.loc[:,stat] + self.conf_sim.loc[:,stat],None,None),
                                     color='b', alpha=.1)
                except:
                    pass
            
        for var in true_values.columns:
            plt.plot(true_values.index, true_values.loc[:,var], label='truevalue_'+var) 
            
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for var in true_values.columns:
            if var in self.stat_names:
                err = self.stai_sim[var] - true_values[var]
                plt.plot(true_values.index, err, label='error_'+var+' a posteriori')
            if var in self.meas_names:
                err = self.outp_sim[var] - true_values[var]
                plt.plot(true_values.index, err, label='error_'+var+' a priori')
             
        plt.legend()
        
        plt.show()
    
    
    def save(self, name='conf_intervals'):
        '''
        Save stored confidence intervals
        
        '''
        
        self.conf_sim.to_csv(name+'.csv')
    
        return self.conf_sim
    
    