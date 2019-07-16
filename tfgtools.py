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
from HRV import *

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


def plot_signal(ecg_signal, fs, seconds):
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



def check_subject(ecg_signal, sampling_rate):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=True)
    plt.figure()
    return out

def get_subject_number(file_route):
    
    subject_number = file_route.split('/')[7]
    #print(subject_number)
    
    return subject_number


#function to perfomr Moving Average (MA) smoothing
def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    #mirroring to take care of the edges
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    
    y=np.convolve(w/w.sum(),s,mode='valid')
    
    y = y[int(np.ceil(window_len/2-1)):-int((window_len/2))] #get same length original signal
    return y

#%% Obtencion de lista de archivos txt con ecg/eda de todos los sujetos

list_txt = get_txt_list()

#f = open("lista_muestras3.txt", "w+")
#f.write(str(list_txt))
#f.close()

print("--------------------------------------------------------")
print(list_txt)
print("--------------------------------------------------------")


#%% Obtención señales hrv (rr_interval) y eda (eda_signal) de un solo sujeto para un anuncio
"""
muestra = 0
txt_file = list_txt[muestra]
print(txt_file)
plt.close('all')
"""
plt.close('all')
#Muestras_erroneas = [2, 4, 7, 8, 12, 13, 22]
#Muestras_raras = [1, 9, 10, 18, 21]
#Directorio 7794 hay dos txt, uno FAILED
#Directorio 8091 no hay txt
#Directorio 8139 hay dos txt, uno no funciona
#Directorio 9570 tiene txt (24) con diferente formato de columnos, no lo lee bien
#txt_file = 'opensignals_79_2019-04-04_12-13-34.txt'

subject_number = 7
#Numero de anuncio, va de 0 a 5
ad_number = 1

txt_file = list_txt[subject_number]
fs = get_sampling_rate(txt_file)

ecg_signal, eda_signal = preprocessing_bitalino_signal(txt_file)
print('ECG: ', ecg_signal)
print('EDA: ', eda_signal)
#plot_ecg_signal(ecg_signal[0], fs, True)


out = check_subject(ecg_signal[ad_number], fs)
plt.show()

## HRV ##
r_peaks = out['rpeaks']

ecg_example = ecg_signal[ad_number]

plt.plot(r_peaks,ecg_example[r_peaks],'rx')  #r_peaks/fs si está en segundos
plot_signal(ecg_signal[ad_number], fs, seconds=False)
plt.title('ECG Ad {}'.format(ad_number))

#subject_number = get_subject_number(txt_file)
#plt.title('Patient {}// Muestra {}'.format(subject_number, ad_number))

plt.figure()
rr_interval = np.diff(r_peaks)
plt.plot(r_peaks[0:-1],rr_interval,'.-')
plt.title('RR_interval Ad {}'.format(ad_number))


## EDA ##
plt.figure()
plot_signal(eda_signal[ad_number], fs, seconds=False)
plt.title('EDA Ad {}'.format(ad_number))


#%% Preprocesado de HRV y EDA

##################
## HRV analysis ##
##################
    
plt.close('all')
#suponemos que estamos analizando un segmento (unos únicos valores de un anuncio)



#tenemos lo siguientes elementos

#r_peaks
#rr_interval
#hr
#hr_ts

#la mayoría los proporciona biosppy y rr_interval lo obtenemos nosotros

#cargamos los datos
#ecg_data = np.load('ecg_data.npz')
#print(ecg_data.files)
rr = rr_interval/fs #rr in secs OJO
#plt.plot(rr)

#corrección de artefactos con HRV
my_hrv = HRV()
prct = 0.2

#creamos lista de labels para los latidos
labels = ['N']*len(rr)

ind_not_N_beats=my_hrv.artifact_ectopic_detection(rr*1000, labels, prct, numBeatsAfterV = 4)
#2. Correction
#if every beat is Normal (sum(ind_not_N_beats) == 0), then no correction
if ind_not_N_beats.sum() > 0:
    rr_corrected = my_hrv.artifact_ectopic_correction(rr, ind_not_N_beats, method='linear')
else:
    rr_corrected = rr.copy()
        
#hr_computation

hr = 60/(rr_corrected)

#MA filtering        
hr_corrected = smooth(hr,window_len = 10)

#smooth rr
rr_smooth = smooth(rr_corrected,window_len=3)

#plot rr
plt.figure()
plt.plot(rr,label = 'RR interval original')
plt.plot(rr_corrected,label = 'RR artifact corrected')
plt.plot(rr_smooth,label = 'RR smoothed')
plt.legend()

plt.figure()
plt.plot(out['heart_rate'],label='HR from biosppy')
plt.plot(hr,label='HR rr_interval corrected')
plt.plot(hr_corrected,label='HR smoothed')
plt.legend()

#%%
##################
## EDA analysis ##
##################

plt.close('all')
"""
siguiendo el paper de KIM 2004 vamos a utilizar los siguientes parámetros:
mean DC level of EDA --> SCL?
mean values of SCR amplitudes
duration of SCR ocurrences
number of ocurrences
"""

#OJO Detected SCRs with an amplitude smaller that 10% of the maximum SCR
#amplitude in this segment were excluded.
#I DON'T KNOW IF THIS IS IMPLEMENTED IN BIOSPPY, CHECK => Checked, it's ok

#load eda data
#eda_raw = np.load('eda.npy')
#plt.plot(eda_raw)
from biosppy.signals import eda as eda_biosppy

#get filtered eda
eda_obj = eda_biosppy.eda(eda_signal[ad_number],sampling_rate = fs,show=False)

#get 
print(eda_obj.keys())


"""
#get amplitudes and peaks using basic_scr
eda_scr = eda_biosppy.basic_scr(eda_obj['filtered'],sampling_rate = fs)
#discard amplitudes samller than 10% maximum amplitude
    
max_amp = np.max(eda_scr['amplitudes'])
thr = 0.1

amp = []
onsets =[]
peaks = []
for i,a in enumerate(eda_scr['amplitudes']):
    
    relative_diff = 1-a/max_amp 
    if relative_diff < thr:
        onsets.append(eda_scr['onsets'][i])
        peaks.append(eda_scr['peaks'][i])
        amp.append(a)
"""

#convert eda filtered (tonic) to microSiemens --> ¿Por qué se convierte así?
Rmohm = 1 - eda_obj['filtered']/2**10
eda = 1/Rmohm


ymin, ymax = np.min(eda),np.max(eda)
alpha = 0.1 * (ymax - ymin)    #Este alpha para qué sirve?? --> para que las 
                               #líneas sobrepasen por encima y por debajo la 
                               #gráfica
ymax += alpha
ymin -= alpha
plt.figure()
plt.plot(eda)
plt.vlines(eda_obj['onsets'], ymin, ymax,color='m',label='Onsets')
plt.vlines(eda_obj['peaks'], ymin, ymax,color='g',label='Peaks')
plt.ylabel('microSiemens')





#%%
    
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