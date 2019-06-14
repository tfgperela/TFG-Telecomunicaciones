#!usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:34:51 2019
@author: rperela
"""

import numpy as np
from biosppy.signals import ecg
import matplotlib.pyplot as plt
import json
import os
#from pathlib import Path
import inspect
import glob

#import sys #, getopt
#Posible utilización de getopt para gestionar sys.argv

def get_txt_list():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    path = (path, '/data')
    
    path = ''.join(path)
    #
    print('PATH: ', path)

    txt_list = []
    #for filename in Path(path).glob('**/*.txt'):
    #    txt_list.append(filename)
    txt_list = glob.glob(path + '/**/*.txt', recursive=True)
    txt_list.sort()
    
    return txt_list


def get_sampling_rate(file):
    
    #Obtenemos la tasa de muestreo de las señales
    with open(file) as f:
        lines = f.readlines()[1]
        line = lines.split(" ", 2)[2]
        line = line[:][0:-2]
        data  = json.loads(line)
        #print(data)
        sampling_rate = data['sampling rate']
        #print(sampling_rate)
        del lines, line, data
        return sampling_rate
    
    

def preprocessing_bitalino_signal(file):

    bit = np.loadtxt(file)
    
    #Cálculo del inicio y final del video
    trigger = bit[:,1]
    trigger_values =  np.where(trigger>0)
    start = trigger_values[0][0]
    #end = trigger_values[0][-1]
    
    #Señales desde el inicio hasta el final del video
    ecg_temp = bit[start:,-2]
    eda_temp = bit[start:,-1]
    #plt.plot(ecg_temp[5000:5200])

    #Señales separadas por anuncios
    time_ad = [0, 60, 120, 180, 226, 287, 347]
    
    ecg_signal = []
    eda_signal = []
    for i in range(6):
        ecg_signal.append(ecg_temp[(time_ad[i]*fs):(time_ad[i+1]*fs)])
        eda_signal.append(eda_temp[(time_ad[i]*fs):(time_ad[i+1]*fs)])
    
    return ecg_signal, eda_signal


def plot_ecg_signal(ecg_signal, fs, seconds):
    if seconds:
        #Gráfico en segundos
        N = len(ecg_signal)  # number of samples
        T = (N - 1) / fs  # duration
        ts = np.linspace(0, T, N, endpoint=False)
    else:
        #Gráfico en muestras
        ts =np.linspace(0, len(ecg_signal), len(ecg_signal), endpoint=False)
        
    plt.plot(ts, ecg_signal, lw=2)
    plt.grid()
    plt.show()



def check_patient(ecg_signal, sampling_rate):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=True)
    plt.figure()
    return out



list_txt = get_txt_list()

#f = open("lista_muestras3.txt", "w+")
#f.write(str(list_txt))
#f.close()

print(list_txt)
print("--------------------------------------------------------")


muestra = 0
txt_file = list_txt[muestra]
print(txt_file)
plt.close('all')


#Muestras_erroneas = [2, 4, 7, 8, 12, 13, 22]
#Muestras_raras = [1, 9, 10, 18, 21]
#Directorio 7794 hay dos txt, uno FAILED
#Directorio 8091 no hay txt
#Directorio 8139 hay dos txt, uno no funciona
#Directorio 9570 tiene txt (24) con diferente formato de columnos, no lo lee bien

fs = get_sampling_rate(txt_file)
ecg_signal, eda_signal = preprocessing_bitalino_signal(txt_file)
print('ECG: ', ecg_signal)
print('EDA: ', eda_signal)
plot_ecg_signal(ecg_signal[0], fs, True)


out = check_patient(ecg_signal[0], fs)
plt.show()
#print(out)
#print(type(out))
r_peaks = out['rpeaks']

ecg_example = ecg_signal[0]
plt.plot(r_peaks,ecg_example[r_peaks],'rx')
plot_ecg_signal(ecg_signal[0], fs, False)
plt.show()



    
#if __name__ == "__main__":
#    if len(sys.argv) == 2:
#        try:
#            ecg, eda = preprocessing_bitalino_signal(sys.argv[1])
#            print('ECG: ', ecg)
#            print('EDA: ', eda)
#        except FileNotFoundError:
#            print ('No such file')
#    else:
#        print('Usage error: tfgtools.py <input_file>')