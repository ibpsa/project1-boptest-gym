'''
Created on Apr 8, 2021

@author: Javier Arroyo

Module to compare references from two different directories. For each
directory, it walks through every file and plots the content of each
pair of files with the same name in the same plot to show the 
differences. 

'''

from testing.utilities import compare_references

vars_timeseries = ['reaTRoo_y', 
                   'reaTZon_y', 
                   'LowerSetp[1]',
                   '0','1','2','3','4','5',]

compare_references(vars_timeseries = vars_timeseries, 
                   refs_old = 'references_old', 
                   refs_new = 'references')
        
        