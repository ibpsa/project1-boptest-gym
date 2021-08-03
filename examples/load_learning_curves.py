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

# Arguments
agents_map = {}
agents_map['DQN_Actual_D_1e06_logdir'] = ['DQN_9', 
                                          'DQN_10']

agents_map['DQN_RC_D_1e06_logdir'] = [
                                    'DQN_1', 
                                    'DQN_2', 
                                    'DQN_3', 
                                    'DQN_4', 
                                    'DQN_5', 
                                     ]

log_dir_parent = os.path.join(utilities.get_root_path(), 'examples', 'agents')
load_from_tb = False
plot = True
linewidth = 0.8
colors    = ['purple', 'darkcyan', 'saddlebrown', 'darkslateblue']
markers   = []

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
            label=agent.split('_')[0]+ ' ' +agent.split('_')[1]
            if 'RC' in agent:
                fillstyle = 'full'
                marker = '^'
            else:
                fillstyle = 'full'
                marker = 'o'
            
            colors[i]='red'
            if label=='DQN RC':
                label = 'DQN (trained in $\mathcal{E}_F$)'
                marker = None
                linestyle = '-'
            elif label=='DQN Actual':
                label = 'DQN (trained in $\mathcal{E}_f$)'
                marker = None
                linestyle = '--'
                            
            plt.plot(df_metric['steps']/1e6, np.array(smoothed)/1e3, color=colors[i], 
                     linestyle=linestyle, linewidth=linewidth, label=label, marker=marker,
                     fillstyle=fillstyle, markevery=50, markersize=3)
    if plot:
        plt.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
        ax.set_xlabel('Million steps')
        ax.set_ylabel('Average episodic return ($10^3$)'.title())
        ax.legend()        
        plt.tight_layout()         
        pdf_file = os.path.join(log_dir_parent, 'metrics', metric.replace('/','_')+'.pdf')       
        plt.savefig(pdf_file, bbox_inches='tight')
        plt.show()



