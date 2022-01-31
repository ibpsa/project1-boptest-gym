'''
Created on Jul 16, 2021

@author: ARROYOJ

This module loads the training data from different 
tensorboard logs, groups it, and plots the learning
curves. It is useful to compare the learning rate
of different RL agents and to collect data of one 
same RL agent that has been interrupted during the
learning process.  

'''

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from testing import utilities
import numpy as np
import pandas as pd
import json
import os
from collections import OrderedDict
import copy
import matplotlib.font_manager as fm
font = fm.FontProperties()

# The following settings are set by default and coincide with KUL Latex template style...
font.set_family('sans-serif') # families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'] 
font.set_fontconfig_pattern('DejaVu Sans') # ['Tahoma', 'Verdana', 'DejaVu Sans', 'Lucia Grande']
font.set_style('normal') # styles = ['normal', 'italic', 'oblique']
fonttitle1 = copy.deepcopy(font)
fonttitle1.set_weight('semibold')

# Arguments
agents_map = OrderedDict()

agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir'] = ['DQN_1']

# agents_map['DQN_Actual_B_Ts15_Th00_1e06_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_C_Ts15_Th03_1e06_logdir'] = ['DQN_1']

# agents_map['DQN_Actual_D_Ts15_Th03_3e05_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th06_3e05_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th12_3e05_logdir'] = ['DQN_1']

agents_map['DQN_Actual_D_Ts30_Th24_3e05_logdir'] = ['DQN_1']
agents_map['DQN_Actual_D_Ts60_Th24_1e06_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_u2'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_Tset'] = ['DQN_1']
# agents_map['SAC_Actual_D_Ts15_Th24_3e05_logdir'] = ['SAC_1']
# agents_map['PPO_Actual_D_Ts15_Th24_3e05_logdir'] = ['PPO2_1']

log_dir_parent = os.path.join(utilities.get_root_path(), 'examples', 'agents')
# log_dir_parent = os.path.join('D:\\','agents')

load_from_tb = False
plot = True
linewidth = 0.8            
max_steps = 0.3e6


# Find directory with RL agents
agnt_dir = os.path.abspath('C:\\Users\\u0110910\\workspace\BOPTEST-gym\\examples\\agents')

all_kpis = pd.DataFrame(columns=['tdis_tot','cost_tot','idis_tot','ener_tot','emis_tot','time_rat'])

for agent in agents_map.keys():
    # Check for prediction horizon
    if '_Th00_' in agent:
        Th = 0
    elif '_Th03_' in agent:
        Th = 10800
    elif '_Th06_' in agent:
        Th = 21600
    elif '_Th12_' in agent:
        Th = 43200
    elif '_Th24_' in agent:
        Th = 86400
    elif '_Th48_' in agent:
        Th = 172800
    # Check for control step period
    if '_Ts15_' in agent:
        Ts = 900
    elif '_Ts30_' in agent:
        Ts = 1800
    elif '_Ts60_' in agent:
        Ts = 3600
        
    # Get learning algorithm
    alg = agent.split('_')[0]
    if '_u2' in agent:
        alg+='_a2'
    if '_TSet' in agent:
        alg+='_TSet'
    
    # Get environment
    env = agent.split('_')[2]
    
    alg_env = alg +'__'+ env
    
    kpis_dic = OrderedDict()
    
    for pricing in ['highly_dynamic']: # 'constant','dynamic',
        # Load peak heating period results
        kpis_dic[alg_env+'__'+pricing+'__'+'peak_heat_day'+'__'+str(Ts)+'__'+str(Th)] = json.load(open(os.path.join(agnt_dir,agent,'results_tests_model_300000_{}'.format(pricing),'kpis_16.json'),'r'))    
        kpis = pd.DataFrame(kpis_dic).T
        all_kpis = pd.concat([all_kpis,kpis],axis=0,sort=True)
        # load typical heating period results
        kpis_dic[alg_env+'__'+pricing+'__'+'typical_heat_day'+'__'+str(Ts)+'__'+str(Th)] = json.load(open(os.path.join(agnt_dir,agent,'results_tests_model_300000_{}'.format(pricing),'kpis_108.json'),'r'))    
        kpis = pd.DataFrame(kpis_dic).T
        all_kpis = pd.concat([all_kpis,kpis],axis=0,sort=True)  


# Print tags of contained entities, use these names to retrieve entities as below
# print(acc.Tags())
# 'scalars':      ['loss/loss', 'input_info/rewards', 'input_info/importance_weights', 'episode_reward', 'loss/td_error']
# 'run_metadata': ['step739999', 'step46099', 'step309699' ...
# In the case of DQN the rewards are the average of the rewards sampled from a batch of the buffer obtained every 100 steps

# https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

metric_tags = [#'loss/loss', 
               #'input_info/rewards', 
               #'input_info/importance_weights', 
               'episode_reward', 
               #'loss/td_error'
               ]
metrics = {}

for metric in metric_tags:
    if plot:
        _, axs = plt.subplots(figsize=(12,4), nrows=1, ncols=3)
    for i,agent in enumerate(agents_map.keys()):
        first_step = 0 
        csv_file = os.path.join(log_dir_parent,agent,metric.replace('/','_')+'.csv')
        columns_metric = ['steps',metric]
        df_metric = pd.DataFrame(columns=columns_metric)
        if load_from_tb:
            for log in agents_map[agent]:
                print('Loading {}'.format(log))
                log_dir = os.path.join(log_dir_parent, agent, log)
                acc = EventAccumulator(log_dir)
                acc.Reload()
                xy = [(s.step, s.value) for s in acc.Scalars(metric)]
                df = pd.DataFrame(xy)
                df.columns = columns_metric
                df['steps'] = df['steps'] + first_step
                df_metric = pd.concat([df_metric,df],axis=0)
                first_step = df['steps'].iloc[-1]
                metrics[metric] = df_metric
            df_metric.to_csv(csv_file, index=False)
        else:
            df_metric = pd.read_csv(csv_file)
            
        if plot:            
            smoothing = 0.99
            # Do not smooth first points which are not representative
            df_metric = df_metric.iloc[2:]
            smoothed = smooth(list(df_metric[metric]), smoothing)

            label = ''
            
            markevery=50
            if 'Ts60' in agent:
                marker = 'p'
                markersize = 6
                label += ' $\Delta t_s=60min$'
                markevery=markevery*4
                color = 'darkorange'
            elif 'Ts30' in agent:
                marker='s'
                markersize = 4
                label += ' $\Delta t_s=30min$'
                markevery=markevery*2
                color = 'darkorange'
            elif 'Ts15' in agent:
                marker = '^'
                markersize = 4
                label += ' $\Delta t_s=15min$'
                markevery=markevery*1
                color = 'darkorange'

            steps = df_metric.loc[df_metric['steps']<max_steps]['steps']
            axs[0].plot(steps/1e6, np.array(smoothed[:len(steps)])/1e3, color=color, 
                     marker=marker, linewidth=linewidth, label=label, 
                     markersize=markersize, markevery=markevery)
            
    plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
    axs[0].set_xlabel('Million steps')
    axs[0].set_ylabel('Average episodic return ($10^3$)')
    axs[0].set_title('Learning curves', fontproperties=fonttitle1)
    axs[0].legend()        

    axs[1].set_xlabel('Total operational cost [EUR/m$^2$]', fontproperties=font)
    axs[1].set_ylabel('Total discomfort [Kh/zone]', fontproperties=font)
    axs[1].set_title('Peak heating period', fontproperties=fonttitle1)
    
    axs[2].set_xlabel('Total operational cost [EUR/m$^2$]', fontproperties=font)
    axs[2].set_ylabel('Total discomfort [Kh/zone]', fontproperties=font)
    axs[2].set_title('Typical heating period', fontproperties=fonttitle1)
    
    plt.tight_layout()         
    pdf_file = os.path.join(log_dir_parent, 'metrics', 'analysis_control_step_period'+'.pdf')       
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.show()



