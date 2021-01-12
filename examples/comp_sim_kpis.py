'''
Created on Jul 17, 2019

@author: Javier Arroyo

'''

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
from collections import OrderedDict


cases = [
         'baseline_feb',
         'baseline_nov',
         'A_feb',
         'A_nov',
         'B_feb',
         'B_nov',
         'C_feb',
         'C_nov'
         ]

kpis_dic = OrderedDict()

for ckey in cases:
    kpis_dic[ckey] = json.load(open('kpis_'+ckey+'.json','r'))
    
kpis = pd.DataFrame(kpis_dic).T
cost_feb = []
cost_nov = []
tdis_feb = []
tdis_nov = []

fig, ax = plt.subplots()

# Plot points
for i, r in kpis.iterrows():
    if 'A' in i:
        color = 'r' 
        label='A'
    elif 'B' in i:
        color = 'b'
        label = 'B'
    elif 'C' in i:
        color = 'c'
        label = 'C'
    elif 'baseline' in i:
        color = 'g'
        label = 'Baseline'

    if '_nov' in i:
        ax.plot(r['cost_tot'], r['tdis_tot'], 'o', markersize=5, linewidth=0.1, label='_nolegend_', color=color)
        cost_nov.append(r['cost_tot'])
        tdis_nov.append(r['tdis_tot'])
    else:
        ax.plot(r['cost_tot'], r['tdis_tot'], 'o', markersize=5, linewidth=0.1, label=label, color=color)
        cost_feb.append(r['cost_tot'])
        tdis_feb.append(r['tdis_tot'])

# Plot lines indicating test case month
points = pd.DataFrame([cost_feb, tdis_feb, cost_nov, tdis_nov], index=['cost_feb', 'tdis_feb', 'cost_nov', 'tdis_nov']).T
points.sort_values(by='cost_feb', axis=0, ascending=False, inplace=True)
ax.plot(list(points['cost_feb']), list(points['tdis_feb']), '--', linewidth=0.8, label='Febuary', color='grey', zorder=1)
ax.plot(list(points['cost_nov']), list(points['tdis_nov']), '-.', linewidth=0.8, label='November', color='grey', zorder=1)

ax.set_xlabel('Total operational cost (EUR)')
ax.set_ylabel('Total discomfort (Kh)')
plt.grid()
plt.legend(loc='best')

# Make comparisons against baseline in percentages
#=====================================================================
# kpis['cost_cmp'] = None
# kpis['tdis_cmp'] = None
# for ckey in cases:
#     if ckey is not 'baseline':
#         kpis['cost_cmp'][ckey] = \
#         (kpis['cost_tot'][ckey]-kpis['cost_tot']['baseline_nov'])/kpis['cost_tot']['baseline_nov']*100
#         kpis['tdis_cmp'][ckey] = \
#         (kpis['tdis_tot'][ckey]-kpis['tdis_tot']['baseline_nov'])/kpis['tdis_tot']['baseline_nov']*100
#=====================================================================

plt.show()
