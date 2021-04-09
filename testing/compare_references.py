'''
Created on Apr 8, 2021

@author: Javier Arroyo

Module to compare references from two different directories. For each
directory, it walks through every file and plots the content of each
pair of files with the same name in the same plot to show the 
differences. 

'''

import utilities
import pandas as pd
import matplotlib.pyplot as plt
import os

vars_timeseries = ['reaTRoo_y', 
                   'reaTZon_y', 
                   'LowerSetp[1]',
                   '0','1','2','3','4','5',]

refs_old = 'references_old'
refs_new = 'references'

dir_old = os.path.join(utilities.get_root_path(), 'testing', refs_old)
dir_new = os.path.join(utilities.get_root_path(), 'testing', refs_new)

for subdir, dirs, files in os.walk(dir_old):
    for filename in files:
        f_old = os.path.join(subdir, filename)
        f_new = os.path.join(subdir.replace(refs_old,refs_new), filename)
        if not os.path.exists(f_new):
            print('File: {} has not been compared since t does not exist anymore.'.format(f_new))
            
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
                                df_old[v].plot(ax=ax, label='old', kind=kind, alpha=0.5, color='orange')
                                df_new[v].plot(ax=ax, label='new', kind=kind, alpha=0.5, color='blue')
                                ax.set_title(str(f_new))
                                ax.legend()
                    else:
                        print('File: {} has not been compared because it does not contain any of the variables to plot'.format(f_old))
            
plt.show()
        
        