#!usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:23:49 2019

@author: rperela
"""

import tfgtools
import sys


#def main(file):
file = 'opensignals_79_2019-04-04_12-13-34.txt'
ecg, eda, trigger = tfgtools.preprocessing_bitalino_signal(file)
    
#print('ECG: ', ecg)
#print('EDA: ', eda)





#if len(sys.argv) == 2:
#    try:
 #       main(sys.argv[1])
 #   except FileNotFoundError:
 #       print ('No such file')
#else:
 #   print('Usage error: tfgtools.py <input_file>')
