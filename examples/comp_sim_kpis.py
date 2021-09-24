'''
Created on Jul 17, 2019

@author: Javier Arroyo

'''

import matplotlib.pyplot as plt
from collections import OrderedDict
from testing import utilities
import pandas as pd
import json
import os
import copy


import matplotlib as mpl
import numpy as np

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='aqua' #blue
c2='red' #green
n=1e6

linewidth=0.6
markersize=3
import matplotlib.font_manager as fm
font = fm.FontProperties()

# The following settings are set by default and coincide with KUL Latex template style...
font.set_family('sans-serif') # families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'] 
font.set_fontconfig_pattern('DejaVu Sans') # ['Tahoma', 'Verdana', 'DejaVu Sans', 'Lucia Grande']
font.set_style('normal') # styles = ['normal', 'italic', 'oblique']
font.set_size(9)

fonttitle1 = copy.deepcopy(font)
fonttitle1.set_weight('semibold')

algorithm = 'DQN'
case      = 'C'
training_timesteps  = 1e6 

# Find log directory
log_dir = os.path.join(utilities.get_root_path(), 'examples', 
                       'agents', '{}_{}_{:.0e}_logdir'.format(algorithm,case,training_timesteps))
log_dir = log_dir.replace('+', '')

model_names = [
    #'model_10000',
    #'model_50000',
    'model_100000',
    'model_200000',
    'model_300000',
    'model_400000',
    'model_500000',
    'model_600000',
    'model_700000',
    'model_800000',
    'model_900000',
    'model_1000000',
    'baseline'
    ]

kpis_dic = OrderedDict()
kpis_dic['peak'] = OrderedDict()
kpis_dic['typi'] = OrderedDict()

for model_name in model_names:
    kpis_dic['peak'][model_name] = json.load(open(os.path.join(log_dir, 
                                    'results_tests_'+model_name, 'kpis_16.json'),'r'))
    kpis_dic['typi'][model_name] = json.load(open(os.path.join(log_dir, 
                                    'results_tests_'+model_name, 'kpis_108.json'),'r'))
    
_, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,3))

for s,scenario in enumerate(kpis_dic.keys()):
    for m,model in enumerate(kpis_dic[scenario].keys()):
        if 'baseline' in model:
            marker='*'
            markersize=5
            color='green'
            label = 'Baseline'
        else:
            training_steps = int(model.split('_')[1])
            marker='o'
            markersize=training_steps/1e6*8+1
            color=colorFader(c1,c2,training_steps/n)
            label=str(training_steps/1e6)+'x1e6'
            
        axs[s].plot(kpis_dic[scenario][model]['cost_tot'], kpis_dic[scenario][model]['tdis_tot'], color=color,
                    marker=marker, markersize=markersize, label=label, linestyle='None')
    axs[s].grid()

#axs[s].legend(loc='upper right', bbox_to_anchor=(1.5, 1), fancybox=True)
plt.legend(loc='upper right', fancybox=True, prop=font) #, bbox_to_anchor=(1.75, 1),

axs[0].set_xlabel('Total operational cost (EUR/m$^2$)', fontproperties=font)
axs[1].set_xlabel('Total operational cost (EUR/m$^2$)', fontproperties=font)

axs[0].set_ylabel('Total discomfort (Kh/zone)', fontproperties=font)

axs[0].set_title('Peak heating period', fontproperties=fonttitle1)
axs[1].set_title('Typical heating period', fontproperties=fonttitle1)

plt.savefig('results_kpi_DDQN_models.pdf', bbox_inches='tight')

plt.show()



