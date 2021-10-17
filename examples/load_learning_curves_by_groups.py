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

agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir'] = 'REF'

agents_map['DQN_Actual_A_Ts15_Th00_3e05_logdir'] = 'SS1'
agents_map['DQN_Actual_B_Ts15_Th00_1e06_logdir'] = 'SS2'
agents_map['DQN_Actual_C_Ts15_Th03_1e06_logdir'] = 'SS3'
agents_map['DQN_Actual_D_Ts15_Th03_3e05_logdir'] = 'SS4'
agents_map['DQN_Actual_D_Ts15_Th06_3e05_logdir'] = 'SS5'
agents_map['DQN_Actual_D_Ts15_Th12_3e05_logdir'] = 'SS6'
agents_map['DQN_Actual_D_Ts30_Th24_3e05_logdir'] = 'SS7'
agents_map['DQN_Actual_D_Ts60_Th24_1e06_logdir'] = 'SS8'

agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_u2']     = 'AS1'
agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_Tset']   = 'AS2'
agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_TsetBkupCon'] = 'AS3'

agents_map['SAC_Actual_D_Ts15_Th24_3e05_logdir'] = 'AL1'
agents_map['A2C_Actual_D_Ts15_Th24_3e05_logdir'] = 'AL2'
agents_map['PPO_Actual_D_Ts15_Th24_3e05_logdir'] = 'AL3'

agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_mlp64'] = 'NN1'
agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_mlp32'] = 'NN2'
agents_map['DQN_Actual_D_Ts15_Th24_3e05_logdir_LnMlp'] = 'NN3'

log_dir_parent = os.path.join(utilities.get_root_path(), 'examples', 'agents')
# log_dir_parent = os.path.join('D:\\','agents')

load_from_tb = False
plot = True
linewidth = 0.8            
max_steps = 0.3e6

color='purple'
color='royalblue'
color='deepskyblue'
color='darkorange'
color='red'
color='darkmagenta'

markers = {}
markers['o'] = 'circle'
markers['s'] = 'square'
markers['D'] = 'diamond'
markers['p'] = 'pentagon'
markers['^'] = 'triangle_up'
markers['>'] = 'triangle_right'
markers['v'] = 'triangle_down'
markers['<'] = 'triangle_left'


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

metric = 'episode_reward'

_, ax = plt.subplots(figsize=(7,6))
for i,agent in enumerate(agents_map.keys()):
    first_step = 0 
    csv_file = os.path.join(log_dir_parent,agent,metric.replace('/','_')+'.csv')
    columns_metric = ['steps',metric]
    df_metric = pd.DataFrame(columns=columns_metric)
    df_metric = pd.read_csv(csv_file)            
    smoothing = 0.99
    # Do not smooth first points which are not representative
    df_metric = df_metric.iloc[2:]
    smoothed = smooth(list(df_metric[metric]), smoothing)
    
    label = agents_map[agent]
    if agents_map[agent] == 'REF':
        color  = 'darkorange'
        marker = 'o'
        
    elif agents_map[agent] == 'SS1':
        color  = 'purple'
        marker = 'o' 
    elif agents_map[agent] == 'SS2':
        color  = 'purple'
        marker = 's' 
    elif agents_map[agent] == 'SS3':
        color  = 'purple'
        marker = 'D' 
    elif agents_map[agent] == 'SS4':
        color  = 'purple'
        marker = 'p' 
    elif agents_map[agent] == 'SS5':
        color  = 'purple'
        marker = '^' 
    elif agents_map[agent] == 'SS6':
        color  = 'purple'
        marker = '>' 
    elif agents_map[agent] == 'SS7':
        color  = 'purple'
        marker = 'v' 
    elif agents_map[agent] == 'SS8':
        color  = 'purple'
        marker = '<' 
    
    elif agents_map[agent] == 'AS1':
        color  = 'deepskyblue'
        marker = 'o' 
    elif agents_map[agent] == 'AS2':
        color  = 'deepskyblue'
        marker = 's' 
    elif agents_map[agent] == 'AS3':
        color  = 'deepskyblue'
        marker = 'D' 
    
    elif agents_map[agent] == 'AL1':
        color  = 'red'
        marker = 'o' 
    elif agents_map[agent] == 'AL2':
        color  = 'red'
        marker = 's' 
    elif agents_map[agent] == 'AL3':
        color  = 'red'
        marker = 'D' 
    
    elif agents_map[agent] == 'NN1':
        color  = 'green'
        marker = 'o' 
    elif agents_map[agent] == 'NN2':
        color  = 'green'
        marker = 's' 
    elif agents_map[agent] == 'NN3':
        color  = 'green'
        marker = 'D' 
    
    markevery=50
    if 'Ts60' in agent:
        markevery=markevery*4
    elif 'Ts30' in agent:
        markevery=markevery*2
    elif 'Ts15' in agent:
        markevery=markevery*1

    markersize = 4
    steps = df_metric.loc[df_metric['steps']<max_steps]['steps']
    plt.plot(steps/1e6, np.array(smoothed[:len(steps)])/1e3, color=color, 
             marker=marker, linewidth=linewidth, label=label, 
             markersize=markersize, markevery=markevery)
plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
ax.set_xlabel('Million steps')
ax.set_ylabel('Average episodic return ($10^3$)')
# ax.legend(loc='right', bbox_to_anchor=(1.15,0.5))        
ax.legend()
plt.tight_layout()         
pdf_file = os.path.join(log_dir_parent, 'metrics', metric.replace('/','_')+'.pdf')       
plt.savefig(pdf_file, bbox_inches='tight')
plt.show()



