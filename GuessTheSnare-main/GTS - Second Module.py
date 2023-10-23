# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:17:16 2023

@author: MuwiHH
"""
#%%

import IPython.display as ipd
import librosa
import librosa.display
#import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scipy
import sklearn
#from sklearn import preprocessing
#import pydub

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import soundfile
import pandas as pd
from minisom import MiniSom
#from adjustText import adjust_text

import os
import math
import glob
from pathlib import Path
import urllib
%matplotlib inline


def data_loading(audio_path):
    data = list(librosa.load(audio_path, sr=22050))#, duration = dur_secs))
    #data = librosa.load(audio_path, sr=22050)#, duration = dur_secs))
    return data

def gate(data):
    gated_data, index = librosa.effects.trim(data[0], top_db=60, frame_length=512, hop_length=256)
    return gated_data

def gate_target(data):
    gated_data, index = librosa.effects.trim(data, top_db=60, frame_length=512, hop_length=256)
    return gated_data

def fourier_db(data, sr):
    signal_fft = np.fft.fft(data)
    freq = np.round(np.fft.rfftfreq(data.size, d= 1/sr))
    signal_fft_abs = np.abs(signal_fft)
    signal_fft_abs = signal_fft_abs[:len(freq)]
    dB = 20 * np.log10(signal_fft_abs / np.max(signal_fft_abs))
    return [freq, dB]

def fourier(data, sr):
    signal_fft = np.fft.fft(data)
    freq = np.round(np.fft.rfftfreq(data.size, d= 1/sr))
    signal_fft_abs = np.abs(signal_fft)
    signal_fft_abs = signal_fft_abs[:len(freq)]
    return [freq, signal_fft_abs]

def fundamental_freqs(stft, indices):
    freqs = []
    for i in range(len(stft)):
        freqs.append(stft[i][0][indices[i]])
    return freqs

def spectral_flux(data, delta: float = 1.0,
                  total: bool = True):
    inp = np.atleast_2d(data).astype('float64')
    out = np.maximum(np.gradient(data, delta, axis=-1), 0)
    if total:
        return out.sum(axis=0, keepdims=True)
    return out

def sc(signal):
    sc = librosa.feature.spectral_centroid(y=signal, n_fft=1024, hop_length=512, sr=signal[1])
    return sc

def zcr(signal):
    zcr = librosa.feature.zero_crossing_rate(y=signal, frame_length=1024, hop_length=512, center=True)
    return zcr

# Minor 2nd filter
f0 = 22.5
cents = range(0, 7200, 100) # until 23KHz
minor2_filter = list(np.round([f0 * (pow((2**((1.0/1200))), i)) for i in cents]))
filter_idx = range(0,52)

def get_filters(fund_freqs):
    filters_list = []
    
    for idx, i in enumerate(fund_freqs):
        first = True  

        for n in range(len(minor2_filter)):
            if first:
                first = False
                continue
            if minor2_filter[n-1] <= i < minor2_filter[n]:
                filters_list.append({'frequency': i, '2m Filter Index': n})
    return filters_list

# Load, gate and normalize target snare sample from loop

gated_target_snare_1 = gate_target(target_snare_1[:2496])
gated_target_snare_1 = gated_target_snare_1.astype(np.float32)

gated_target_snare_2 = gate_target(target_snare_2)
gated_target_snare_2 = gated_target_snare_2.astype(np.float32)

gated_target_snare_3 = gate_target(target_snare_3)
gated_target_snare_3 = gated_target_snare_3.astype(np.float32)

loaded_data = []

# Apply -60dB gate to each audio file and store the gated segments
for audio_path in sorted(Path().glob("Desktop/MARTINEZ DATA SET/Guess The Snare/DATA SET/Snare*.wav")):
    audio_data = data_loading(audio_path)
    loaded_data.append(audio_data)

gated_data_segments = [gate(data) for data in loaded_data]

# append target snares
gated_data_segments.append(gated_target_snare_1)
gated_data_segments.append(gated_target_snare_2)
gated_data_segments.append(gated_target_snare_3)

# 2.2.2. Feature Extraction of the EB

# Feature 1: ZCR
zero_crossings = [np.median(zcr(i)) for i in gated_data_segments]

# Feature 2: STFT
stft = [fourier_db(i, sr=22050) for i in gated_data_segments]

#extract the index of the np.max and filter frequencies
indices = [np.where(i[1] == 0)[0][0] for i in stft]
fund_freqs = fundamental_freqs(stft, indices)
filtered_fund_freqs = get_filters(fund_freqs)

# Feature 3: Spectral Centroid
spectral_c = [np.median(sc(data)) for data in gated_data_segments]

# 2.2.3. Clustering with K-Means (KM)

# Combine the feature lists into a feature matrix for ZCR and SC
feature_matrix = np.column_stack((zero_crossings, spectral_c))
# Normalization of Data, scale from -1 to 1

min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(feature_matrix)

# develop a model to find 3 clusters

model = cluster.KMeans(n_clusters=3) 
labels = model.fit_predict(features_scaled) # fit predict 

plt.figure(figsize=(6, 6))
plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b', marker=".")
plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r', marker=".")
plt.scatter(features_scaled[labels==2,0], features_scaled[labels==2,1], c='y', marker=".")

# add sample idx numbers
texts = [plt.text(features_scaled[i,0], features_scaled[i,1], f'{i}') for i in range(features_scaled.shape[0])]
adjust_text(texts, arrowprops={'arrowstyle':'-', 'color':'green'})

plt.xlabel('ZCR (scaled)')
plt.ylabel('Spectral Centroid (scaled)')
plt.legend(('Cluster 1', 'Cluster 2', 'Cluster 3'))
plt.savefig('k-means-zcr-sc.png', bbox_inches = 'tight')
plt.show()

# for ZCR and frequency
feature_matrix_2 = np.column_stack((zero_crossings, [i['frequency'] for i in filtered_fund_freqs]))#, [i['Filter Index'] for i in filtered_fund_freqs]
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled_2 = min_max_scaler.fit_transform(feature_matrix_2)
model = cluster.KMeans(n_clusters=3) 
labels = model.fit_predict(features_scaled) # fit predict 
plt.figure(figsize=(6, 6))
plt.scatter(features_scaled_2[labels==0,0], features_scaled_2[labels==0,1], c='b', marker=".")
plt.scatter(features_scaled_2[labels==1,0], features_scaled_2[labels==1,1], c='r', marker=".")
plt.scatter(features_scaled_2[labels==2,0], features_scaled_2[labels==2,1], c='y', marker=".")
texts = [plt.text(features_scaled_2[i,0], features_scaled_2[i,1], f'{i}') for i in range(features_scaled_2.shape[0])]
adjust_text(texts, arrowprops={'arrowstyle':'-', 'color':'green'})

plt.xlabel('ZCR (scaled)')
plt.ylabel('Frequencies (scaled)')
plt.legend(('Cluster 1', 'Cluster 2', 'Cluster 3'))
plt.savefig('k-means-zcr-freq.png', bbox_inches = 'tight')
plt.show()

# for SC and frequencies

feature_matrix_3 = np.column_stack((spectral_c, [i['frequency'] for i in filtered_fund_freqs]))#, [i['Filter Index'] for i in filtered_fund_freqs]
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled_3 = min_max_scaler.fit_transform(feature_matrix_3)
model = cluster.KMeans(n_clusters=3) 
labels = model.fit_predict(features_scaled_3)

plt.figure(figsize=(6, 6))
plt.scatter(features_scaled_3[labels==0,0], features_scaled_3[labels==0,1], c='b', marker=".")
plt.scatter(features_scaled_3[labels==1,0], features_scaled_3[labels==1,1], c='r', marker=".")
plt.scatter(features_scaled_3[labels==2,0], features_scaled_3[labels==2,1], c='y', marker=".")

texts = [plt.text(features_scaled_3[i,0], features_scaled_3[i,1], f'{i}') for i in range(features_scaled_3.shape[0])]
adjust_text(texts, arrowprops={'arrowstyle':'-', 'color':'green'})

plt.xlabel('Spectral Centroid (scaled)')
plt.ylabel('Frequencies (scaled)')
plt.legend(('Cluster 1', 'Cluster 2', 'Cluster 3'))
plt.savefig('k-means-sc-freq.png', bbox_inches = 'tight')
plt.show()

#%%

a = [1,2,4,32,38,45,46,47]


data = []
for i in a:
    d = [i, zero_crossings[i], fund_freqs[i], spectral_c[i], features_scaled[i][0], features_scaled_2[i][1], features_scaled[i][1]]
    data.append(d)

path = 'C:/Users/MuwiHH/Desktop/MARTINEZ DATA SET/Code/data_final.csv'
df_data = pd.DataFrame(data)
df_data.to_csv(path, index=False)
print("Done")