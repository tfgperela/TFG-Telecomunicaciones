#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:53:50 2019

@author: obarquero
"""
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from HRV import *

#%%
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

#%%
    
plt.close('all')
#suponemos que estamos analizando un segmento (unos únicos valores de un anuncio)



#tenemos lo siguientes elementos

#r_peaks
#rr_interval
#hr
#hr_ts

#la mayoría los proporciona biosppy y rr_interval lo obtenemos nosotros

#cargamos los datos
ecg_data = np.load('ecg_data.npz')
print(ecg_data.files)
rr = ecg_data['rr_interval'] #rr in secs OJO

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
plt.plot(ecg_data['hr'],label='HR from biosppy')
plt.plot(hr,label='HR rr_interval corrected')
plt.plot(hr_corrected,label='HR smoothed')
plt.legend()


#%% EDA analysis

"""
siguiendo el paper de KIM 2004 vamos a utilizar los siguientes parámetros:
mean DC level of EDA
mean values of SCR amplitudes
duration of SCR ocurrences
number of ocurrences
"""

#OJO Detected SCRs with an amplitude smaller that 10% of the maximum SCR
#amplitude in this segment were excluded.
#I DON'T KNOW IF THIS IS IMPLEMENTED IN BIOSPPY, CHECK => Checked, it's ok

#load eda data
eda_raw = np.load('eda.npy')
from biosppy.signals import eda as eda_biosppy

fs = 100

#get filtered eda
eda_obj = eda_biosppy.eda(eda_raw,sampling_rate = fs,show=False)

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

#convert eda filtered (tonic) to microSiemens
Rmohm = 1 - eda_obj['filtered']/2**10
eda = 1/Rmohm


ymin, ymax = np.min(eda),np.max(eda)
alpha = 0.1 * (ymax - ymin)
ymax += alpha
ymin -= alpha
plt.figure()
plt.plot(eda)
plt.vlines(eda_obj['onsets'], ymin, ymax,color='m',label='Onsets')
plt.vlines(eda_obj['peaks'], ymin, ymax,color='g',label='Peaks')
plt.ylabel('micro iemens')

#nota: en eda_obj['amplitudes'] se encuentran las amplitudes

