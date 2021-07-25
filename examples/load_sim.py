'''
Created on Jul 25, 2021

@author: ARROYOJ

'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

from collections import OrderedDict
from examples.test_and_plot import reindex, create_datetime

def plot_results(res_lists):
    
    data = {}
    temper = {}
    rewtog = {}
    # Load the data
    for label, res_list in res_lists.items():
        data[label] = {}
        data[label]['peak'] = pd.read_csv(os.path.join(res_list[0],'results_sim_16.csv'),  index_col='datetime')
        data[label]['typi'] = pd.read_csv(os.path.join(res_list[0],'results_sim_108.csv'), index_col='datetime')
        if 'RLMPC' in label:
            temper[label] = {}
            rewtog[label] = {}
            temper[label]['peak'] = pd.read_json(os.path.join(res_list[0],'temper_16.json'))
            temper[label]['typi'] = pd.read_json(os.path.join(res_list[0],'temper_108.json'))
            rewtog[label]['peak'] = pd.read_json(os.path.join(res_list[0],'rewtog_16.json'))
            rewtog[label]['typi'] = pd.read_json(os.path.join(res_list[0],'rewtog_108.json'))
        
    # Plot the results
    _, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(10,8))
    
    # Use last results to obtain x indexes
    x_time_peak = pd.to_datetime(data[label]['peak'].index) 
    x_time_typi = pd.to_datetime(data[label]['typi'].index)
    x_time      = [x_time_peak, x_time_typi]
    
    for label, res_list in res_lists.items():
        for p,period in enumerate(['peak', 'typi']):
            df = data[label][period]
            axs[0,p].plot(x_time[p], data[label][period]['reaTZon_y']    - 273.15, color=res_list[1], linewidth=1, label=label, 
                          marker=res_list[2], markevery=500, markersize=3)
            axs[0,p].plot(x_time[p], df['LowerSetp[1]'] - 273.15, color='gray', linewidth=1, label='_nolegend_')
            axs[0,p].plot(x_time[p], df['UpperSetp[1]'] - 273.15, color='gray', linewidth=1, label='_nolegend_')
            
            if 'RLMPC' in label and False:
                Z = None
                axs[0,p].imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn, # hot
                                origin='lower', extent=[-3, 3, -3, 3],
                                vmax=Z.max(), vmin=Z.min())

    for p, period in enumerate(['peak', 'typi']):
        df = data[label][period]
        axs[0,p].set_yticks(np.arange(15, 41, 5))
        axt0 = axs[0,p].twinx()
        axt0.plot(x_time[p], df['PriceElectricPowerHighlyDynamic'], color='dimgray', linestyle='dotted', linewidth=1, label='_nolegend_')
        axs[0,p].plot([],[], color='dimgray', linestyle='-', linewidth=1, label='_nolegend_')
        axt0.set_ylim(0,0.3)
        axt0.set_yticks([])
     
        axs[1,p].plot(x_time[p], df['TDryBul'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
        axs[1,p].set_yticks(np.arange(-5, 26, 5))
        axt1 = axs[1,p].twinx()
        axt1.plot(x_time[p], df['HDirNor'], color='gold', linestyle='-', linewidth=1, label='_nolegend_')
        axt1.set_ylim(0,800)
        axt1.set_yticks([])
        
        axs[1,p].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
     
            
    axs[0,0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
    axs[1,0].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
    axt0.set_ylabel('Price\n(EUR/kWh)')        
    axt1.set_ylabel('Solar\nirradiation\n($W/m^2$)')
    axt0.set_yticks(np.arange(0, 0.31, 0.1))
    axt1.set_yticks(np.arange(0, 801, 100))
    
    axs[0,1].plot([],[], color='gray',                              linewidth=1, label='Comfort setp.')
    axs[0,1].plot([],[], color='dimgray',     linestyle='dotted',   linewidth=1, label='Price')
    axs[1,1].plot([],[], color='royalblue',   linestyle='-',        linewidth=1, label='$T_a$                 ') # The blanks are added to use the same values of bbox_to_anchor
    axs[1,1].plot([],[], color='gold',        linestyle='-',        linewidth=1, label='$\dot{Q}_{rad}$')
    axs[0,1].legend(fancybox=True, bbox_to_anchor=(1.5, 0.98))
    axs[1,1].legend(fancybox=True, bbox_to_anchor=(1.5, 0.98))
    # -0.16
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.94)
    plt.savefig('sim_all.pdf', bbox_inches='tight')     
    plt.show()  

if __name__=='__main__':
    res_lists = OrderedDict()
    #'RL Actual':    ['C:\\workspace\\BOPTEST-gym\\examples\\agents\\DQN_Actual_D_1e06_logdir\\results_tests\\DQN_trained_with_Actual__and__tested_with_Actual'],
    res_lists['DQN']    = ['C:\\workspace\\BOPTEST-gym\\examples\\agents\\DQN_RC_D_1e06_logdir\\results_tests\\DQN_trained_with_RC__and__tested_in_Actual', 'red', '^']
    #'RLMPC Actual': ['C:\\workspace\\BOPTEST-gym\\examples\\agents\\DQN_Actual_D_1e06_logdir\\results_tests\\RLMPC_trained_with_Actual__and__tested_with_Actual_nactions_3'],
    res_lists['RLMPC']  = ['C:\\workspace\\BOPTEST-gym\\examples\\agents\\DQN_RC_D_1e06_logdir\\results_tests\\RLMPC_trained_with_RC__and__tested_in_Actual_nactions11', 'green', 'o']
    
    plot_results(res_lists)
