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
    ]

kpis_dic = OrderedDict()
kpis_dic['peak'] = OrderedDict()
kpis_dic['typi'] = OrderedDict()

for model_name in model_names:
    kpis_dic['peak'][model_name] = json.load(open(os.path.join(log_dir, 
                                    'results_tests_'+model_name, 'kpis_16.json'),'r'))
    kpis_dic['typi'][model_name] = json.load(open(os.path.join(log_dir, 
                                    'results_tests_'+model_name, 'kpis_108.json'),'r'))
    
_, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,4))

for s,scenario in enumerate(kpis_dic.keys()):
    for m,model in enumerate(kpis_dic[scenario].keys()):
        axs[s].plot(kpis_dic[scenario][model]['cost_tot'], kpis_dic[scenario][model]['tdis_tot'],
                    marker='o', markersize=4, label=model, linewidth=0.1)
    axs[s].grid()

axs[s].legend(loc='upper right', bbox_to_anchor=(1.5, 1), fancybox=True)

plt.subplots_adjust(right=0.9, top=0.94)

plt.tight_layout()

plt.savefig('results_kpi_DDQN_models.pdf', bbox_inches='tight')

plt.show()



