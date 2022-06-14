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
import os
from collections import OrderedDict

# Arguments
agents_map = OrderedDict()

# agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir'] = ['DQN_1']

# agents_map['DQN_Actual_D_Ts15_Th03_3e05_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th06_3e05_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th12_3e05_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts30_Th24_3e05_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts60_Th24_1e06_logdir'] = ['DQN_1']

# agents_map['DQN_Actual_A_Ts15_Th00_3e05_logdir'] = ['DQN_2']
# agents_map['DQN_Actual_B_Ts15_Th00_1e06_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_C_Ts15_Th03_1e06_logdir'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_u2'] = ['DQN_1']
# agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_Tset'] = ['DQN_1']

# agents_map['SAC_Actual_D_Ts15_Th24_3e05_logdir'] = ['SAC_1']
# agents_map['PPO_Actual_D_Ts15_Th24_3e05_logdir'] = ['PPO2_1']

# agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_mlp64'] = ['DQN_4']
agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_mlp32'] = ['DQN_1']

log_dir_parent = os.path.join(utilities.get_root_path(), 'examples', 'agents')
# log_dir_parent = os.path.join('D:\\','agents')

load_from_tb = True
plot = True
linewidth = 0.8            
max_steps = 0.3e6


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
        _, ax = plt.subplots(figsize=(5,4))
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
            
            if 'DQN' in agent:
                label += 'DDQN'
            elif 'SAC' in agent:
                label += 'SAC'
            elif 'PPO' in agent:
                label += 'PPO'
            elif 'A2C' in agent:
                label += 'A2C'
            else:
                label = 'nan Agent'
            
            
            if '_Th03_' in agent:
                color='green'
                label += ' $\Delta t_h=03h$'
            elif '_Th06_' in agent:
                color='aquamarine'
                label += ' $\Delta t_h=06h$'
            elif '_Th12_' in agent:
                color='deepskyblue'
                label += ' $\Delta t_h=12h$'
            elif '_Th24_' in agent:
                color='darkorange'
                label += ' $\Delta t_h=24h$'
            elif '_Th48_' in agent:
                color='red'
                label += ' $\Delta t_h=48h$'
            
            markevery=50
            if '_\Delta t_s60_' in agent:
                marker = 'p'
                markersize = 6
                label += ' $T_s=60min$'
                markevery=markevery*4
            elif '_\Delta t_s30_' in agent:
                marker='s'
                markersize = 4
                label += ' $T_s=30min$'
                markevery=markevery*2
            elif '_\Delta t_s15_' in agent:
                marker = '^'
                markersize = 4
                label += ' $T_s=15min$'
                markevery=markevery*1

            if '_A_' in agent:
                label += ' A'
                color = 'red'
            elif '_B_' in agent:
                label += ' B'
                color = 'violet'
            elif '_C_' in agent:
                label += ' C'
                color = 'purple'
            elif '_D_' in agent:
                label += ' D'
                color = 'red'
            else:
                label += 'nan env'
            
            marker = 'o'
            markersize = 4
            steps = df_metric.loc[df_metric['steps']<max_steps]['steps']
            plt.plot(steps/1e6, np.array(smoothed[:len(steps)])/1e3, color=color, 
                     marker=marker, linewidth=linewidth, label=label, 
                     markersize=markersize, markevery=markevery)
    if plot:
        plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
        ax.set_xlabel('Million steps')
        ax.set_ylabel('Average episodic return ($10^3$)')
        ax.legend()        
        plt.tight_layout()         
        pdf_file = os.path.join(log_dir_parent, 'metrics', metric.replace('/','_')+'.pdf')       
        plt.savefig(pdf_file, bbox_inches='tight')
        plt.show()



