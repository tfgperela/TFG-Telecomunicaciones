#!usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:34:51 2019

@author: rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import json



def preprocessing_bitalino_signal(file):

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
    
    
    
    bit = np.loadtxt(file)
    
    #Cálculo del inicio y final del video
    trigger = bit[:,1]
    trigger_values =  np.where(trigger>0)
    start = trigger_values[0][0]
    end = trigger_values[0][-1]
    
    #Señales desde el inicio hasta el final del video
    ecg_temp = bit[start:end,-2]
    eda_temp = bit[start:end,-1]
    #plt.plot(ecg_temp[5000:5200])

    #Señales separadas por anuncios
    time_ad = [0, 60, 120, 180, 226, 287, 347]
    
    ecg = []
    eda = []
    for i in range(6):
        ecg.append(ecg_temp[(time_ad[i]*sampling_rate):(time_ad[i+1]*sampling_rate)])
        eda.append(eda_temp[(time_ad[i]*sampling_rate):(time_ad[i+1]*sampling_rate)])
        
    #PREGUNTA --> ecg y eda son 347000 muestras (por la duración del video) mientras que
    #ecg_temp y eda_temp son 34944 muestras (por el cálculo hecho del trigger), 244 muestras
    #de diferencia. ¿Influye en algo?¿Hay que hacerlo cuadrar?
    
    return ecg, eda
    
    
    
ecg, eda = preprocessing_bitalino_signal("opensignals_79_2019-04-04_12-13-34.txt")

print('ECG: ', ecg)
print('EDA: ', eda)
