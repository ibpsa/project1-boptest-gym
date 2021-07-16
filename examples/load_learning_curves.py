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
import tensorboard as tb
from testing import utilities
import numpy as np
import pandas as pd
import os

# Arguments
agents_map = {}
agents_map['DQN_D_1e06_logdir'] = ['DQN_36', 
                                   'DQN_47', 
                                   'DQN_48', 
                                   'DQN_49', 
                                   'DQN_50']
log_dir_parent = os.path.join(utilities.get_root_path(), 'examples', 'agents')
load_from_tb = False
plot = True

# Print tags of contained entities, use these names to retrieve entities as below
# print(acc.Tags())
first_step = 0 
metric_tags = [#'loss/loss', 
               'input_info/rewards', 
               #'input_info/importance_weights', 
               #'episode_reward', 
               #'loss/td_error'
               ]
metrics = {}

for metric in metric_tags:
    columns_metric = ['steps',metric]
    df_metric = pd.DataFrame(columns=columns_metric)
    for agent,agent_logs in agents_map.items():
        csv_file = os.path.join(log_dir_parent,'metrics',agent+'__'+metric.replace('/','_')+'.csv')
        if load_from_tb:
            for log in agent_logs:
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
        smoothing = 0.
        
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
        
        _, ax = plt.subplots()
        smoothed = smooth(list(df_metric[metric]), smoothing)
        plt.plot(df_metric['steps']/1e6,smoothed)
        ax.set_xlabel('Million steps')
        ax.set_ylabel(metric)
        plt.show()
        



# 'scalars':      ['loss/loss', 'input_info/rewards', 'input_info/importance_weights', 'episode_reward', 'loss/td_error']
# 'run_metadata': ['step739999', 'step46099', 'step309699' ...
# the rewards are the average of the rewards sampled from a batch of the buffer obtained every 100 steps
#===============================================================================
# # Retrieve training reward
# x, y = ts2xy(load_results(log_dir), 'timesteps')
# if len(x) > 0:
#     # Mean training reward over the last self.check_freq episodes
#     mean_reward = np.mean(y[-self.check_freq:])
#     if self.verbose > 0:
#         print("Num timesteps: {}".format(self.num_timesteps))
#         print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
#===============================================================================


