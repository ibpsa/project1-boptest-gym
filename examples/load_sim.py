'''
Created on Jul 25, 2021

@author: ARROYOJ

'''

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from collections import OrderedDict

from testing import utilities
from examples.test_and_plot import reindex, create_datetime

linewidth=0.6
markersize=3
import matplotlib.font_manager as fm
font = fm.FontProperties()

# The following settings are set by default and coincide with KUL Latex template style...
font.set_family('sans-serif') # families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'] 
font.set_fontconfig_pattern('DejaVu Sans') # ['Tahoma', 'Verdana', 'DejaVu Sans', 'Lucia Grande']
font.set_style('normal') # styles = ['normal', 'italic', 'oblique']

fonttitle1 = copy.deepcopy(font)
fonttitle1.set_weight('semibold')

def plot_results(res_lists):
    
    data = {}
    kpis = {}
    temper = {}
    rewtog = {}
    # Load the data
    for label, res_list in res_lists.items():
        if 'RLMPC' in label:
            data[label]   = {}
            kpis[label]   = {}
            temper[label] = {}
            rewtog[label] = {}
            data[label]['peak'] = pd.read_csv(os.path.join(res_list[0],'results_sim_16.csv'),  index_col='datetime')
            data[label]['typi'] = pd.read_csv(os.path.join(res_list[0],'results_sim_108.csv'), index_col='datetime')
            kpis[label]['peak'] = pd.read_json(os.path.join(res_list[0],'kpis_16.json'),typ='series')
            kpis[label]['typi'] = pd.read_json(os.path.join(res_list[0],'kpis_108.json'),typ='series')
            temper[label]['peak'] = pd.read_json(os.path.join(res_list[0],'temper_16.json'))
            temper[label]['typi'] = pd.read_json(os.path.join(res_list[0],'temper_108.json'))
            rewtog[label]['peak'] = pd.read_json(os.path.join(res_list[0],'rewtog_16.json'))
            rewtog[label]['typi'] = pd.read_json(os.path.join(res_list[0],'rewtog_108.json'))
        elif 'MPC' in label:
            data[label] = {}
            kpis[label] = {}
            data[label]['peak'] = pd.read_csv(os.path.join(res_list[0],'mpc__highly_dynamic__peak_heat_day__900__86400','plant.csv'), index_col='datetime').iloc[:-1]
            data[label]['typi'] = pd.read_csv(os.path.join(res_list[0],'mpc__highly_dynamic__typical_heat_day__900__86400','plant.csv'), index_col='datetime').iloc[:-1]
            kpis[label]['peak'] = pd.read_json(os.path.join(res_list[0],'mpc__highly_dynamic__peak_heat_day__900__86400','kpis.json'),typ='series')
            kpis[label]['typi'] = pd.read_json(os.path.join(res_list[0],'mpc__highly_dynamic__typical_heat_day__900__86400','kpis.json'),typ='series')
        else:
            data[label] = {}
            kpis[label] = {}
            data[label]['peak'] = pd.read_csv(os.path.join(res_list[0],'results_sim_16.csv'),  index_col='datetime')
            data[label]['typi'] = pd.read_csv(os.path.join(res_list[0],'results_sim_108.csv'), index_col='datetime')
            kpis[label]['peak'] = pd.read_json(os.path.join(res_list[0],'kpis_16.json'),typ='series')
            kpis[label]['typi'] = pd.read_json(os.path.join(res_list[0],'kpis_108.json'),typ='series')
            
    # Use last results to obtain x indexes and setpoints
    x_time_peak = pd.to_datetime(data[label]['peak'].index) 
    x_time_typi = pd.to_datetime(data[label]['typi'].index)
    x_time      = [x_time_peak, x_time_typi]
    
    if False:
        x_time_step_peak = pd.date_range(start=x_time_peak[0], 
                                         end  =x_time_peak[-1], 
                                         freq ='900s')
        x_time_step_typi = pd.date_range(start=x_time_typi[0], 
                                         end  =x_time_typi[-1], 
                                         freq ='900s')
        x_time_step = [x_time_step_peak, x_time_step_typi]

    lower_setp = {}
    upper_setp = {}
    for period in ['peak', 'typi']:
        df = data[label][period]
        lower_setp[period] = df['LowerSetp[1]'] - 273.15
        upper_setp[period] = df['UpperSetp[1]'] - 273.15
    
    # Plot the simulation results
    _, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(9,6))
    
    for label, res_list in res_lists.items():
        for p,period in enumerate(['peak', 'typi']):
            df = data[label][period]
            axs[0,p].plot(x_time[p], data[label][period]['reaTZon_y']    - 273.15, color=res_list[1], linestyle=res_list[3], linewidth=linewidth, label='_nolegend_', 
                          marker=res_list[2], markevery=5000, markersize=markersize)
            axs[0,p].plot(x_time[p], lower_setp[period], color='gray', linewidth=linewidth, label='_nolegend_')
            axs[0,p].plot(x_time[p], upper_setp[period], color='gray', linewidth=linewidth, label='_nolegend_')
            
            if 'RLMPC' in label and False:
                axs[0,p].plot(x_time_step[p], temper[label][period])
                Z = None
                axs[0,p].imshow(Z, interpolation='bilinear',
                                origin='lower', extent=[-3, 3, -3, 3],
                                vmax=Z.max(), vmin=Z.min())

    for p, period in enumerate(['peak', 'typi']):
        df = data[label][period]
        axs[0,p].set_yticks(np.arange(15, 41, 5))
        axt0 = axs[0,p].twinx()
        axt0.plot(x_time[p], df['PriceElectricPowerHighlyDynamic'], color='dimgray', linestyle='dotted', linewidth=linewidth, label='_nolegend_')
        axs[0,p].plot([],[], color='dimgray', linestyle='-', linewidth=linewidth, label='_nolegend_')
        axt0.set_ylim(0,0.3)
        axt0.set_yticks([])
     
        axs[1,p].plot(x_time[p], df['TDryBul'] - 273.15, color='royalblue', linestyle='-', linewidth=linewidth, label='_nolegend_')
        axs[1,p].set_yticks(np.arange(-5, 26, 5))
        axt1 = axs[1,p].twinx()
        axt1.plot(x_time[p], df['HDirNor'], color='gold', linestyle='-', linewidth=linewidth, label='_nolegend_')
        axt1.set_ylim(0,800)
        axt1.set_yticks([])
        
        axs[1,p].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
     
            
    axs[0,0].set_ylabel('Operative temperature ($^\circ$C)')
    axs[1,0].set_ylabel('Ambient temperature ($^\circ$C)')
    axt0.set_ylabel('Price (EUR/kWh)')        
    axt1.set_ylabel('Solar irradiation ($W/m^2$)')
    axt0.set_yticks(np.arange(0, 0.31, 0.1))
    axt1.set_yticks(np.arange(0, 801, 100))
    
    plt.subplots_adjust(right=0.9, top=0.94)
    
    axs[0,0].set_title('Peak heating period', fontproperties=fonttitle1)
    axs[0,1].set_title('Typical heating period', fontproperties=fonttitle1)
    
    plt.tight_layout()

    for label, res_list in res_lists.items():
        axs[1,1].plot([], [], color=res_list[1], linestyle=res_list[3], linewidth=linewidth, label=label, 
              marker=res_list[2], markersize=markersize)    
    axs[1,1].plot([],[], color='gray',                              linewidth=linewidth, label='Comfort setp.')
    axs[1,1].plot([],[], color='dimgray',     linestyle='dotted',   linewidth=linewidth, label='Price')
    axs[1,1].plot([],[], color='royalblue',   linestyle='-',        linewidth=linewidth, label='$T_a$') 
    axs[1,1].plot([],[], color='gold',        linestyle='-',        linewidth=linewidth, label='$\dot{Q}_{rad}$')
    axs[1,1].legend(fancybox=True, ncol=4, bbox_to_anchor=(0.9, -0.2)) 

    plt.savefig('sim_all.pdf', bbox_inches='tight')     
    plt.show()  

    # Plot the KPIs
    _, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,3))
    
    for label, res_list in res_lists.items():
        if 'DQN (trained in $\mathcal{E}_F$)' not in label:
            for p,period in enumerate(['peak', 'typi']):
                df = kpis[label][period]
                axs[p].plot(df['cost_tot'], df['tdis_tot'], color=res_list[1], linewidth=linewidth, label=label, 
                              marker=res_list[2], markersize=markersize, linestyle="None")
    axs[0].grid()
    axs[1].grid()
    
    plt.tight_layout()
    plt.legend(loc='upper right', fancybox=True, prop=font) #, bbox_to_anchor=(1.75, 1),
    
    axs[0].set_xlabel('Total operational cost (EUR/m$^2$)', fontproperties=font)
    axs[1].set_xlabel('Total operational cost (EUR/m$^2$)', fontproperties=font)
    
    axs[0].set_ylabel('Total discomfort (Kh/zone)', fontproperties=font)
    
    axs[0].set_title('Peak heating period', fontproperties=fonttitle1)
    axs[1].set_title('Typical heating period', fontproperties=fonttitle1)
    
    plt.savefig('kpis_MPCvsRL.pdf', bbox_inches='tight')     
    plt.show()  


if __name__=='__main__':
    res_lists = OrderedDict()
    agents_dir = os.path.join(utilities.get_root_path(), 'examples','agents')
    res_lists['MPC'] = [os.path.join(os.path.dirname(utilities.get_root_path()), 'BOPTEST-control', 'RSH_HP','experiments'),
                       'darkorange', 'D', '-']
    res_lists['DQN (trained in $\mathcal{E}_F$)']    = [os.path.join(agents_dir, 'DQN_RC_D_1e06_logdir', 'results_tests', 'DQN_trained_with_RC__and__tested_in_Actual'), 
                           'red', '<', '-']
    res_lists['DQN (trained in $\mathcal{E}_f$)'] = [os.path.join(agents_dir, 'DQN_Actual_D_1e06_logdir', 'results_tests', 'DQN_trained_with_Actual__and__tested_in_Actual'), 
                           'red', '>', '--']
    res_lists['RLMPC ($\gamma=0.99)$']  = [os.path.join(agents_dir, 'DQN_RC_D_1e06_logdir', 'results_tests', 'RLMPC_trained_with_RC__and__tested_in_Actual_nactions11_w1e6'),
                           'green', '^', '-']
    res_lists['RLMPC ($\gamma=0$)']  = [os.path.join(agents_dir, 'DQN_RC_D_1e06_logdir', 'results_tests', 'RLMPC_trained_with_RC__and__tested_in_Actual_nactions11_gamma0_w1e6'),
                           'green', 'v', '--']
    
    plot_results(res_lists)
