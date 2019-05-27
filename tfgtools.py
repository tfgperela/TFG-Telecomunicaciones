#!usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:34:51 2019

@author: rperela
"""

import numpy as np
import matplotlib.pyplot as plt
import json



#def preprocessing_bitalino_signal(fichero):

#Obtenemos la tasa de muestreo de las señales
with open("opensignals_79_2019-04-04_12-13-34.txt") as f:
    lineas = f.readlines()[1]
    linea = lineas.split(" ", 2)[2]
    linea = linea[:][0:-2]
    #print(linea)
    #print("----------------------------")
    data  = json.loads(linea)
    #print(data)
    sampling_rate = data['sampling rate']
    print(sampling_rate)
    del lineas, linea, data



bit = np.loadtxt("opensignals_79_2019-04-04_12-13-34.txt")

#Cálculo del inicio y final del video
trigger = bit[:,1]
trigger_values =  np.where(trigger>0)
inicio = trigger_values[0][0]
fin = trigger_values[0][-1]



ecg = bit[inicio:fin,-2]
eda = bit[inicio:fin,-1]
    
plt.plot(ecg[5000:5200])
    
   
#return bit, ecg, eda
    
    
    
    
#bit, ecg, eda = prueba("opensignals_79_2019-04-04_12-13-34.txt")
