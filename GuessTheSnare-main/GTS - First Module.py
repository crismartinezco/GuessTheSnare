# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:07:53 2023

@author: MuwiHH
"""

# 2.2. EXTRACTION OF THE SNARE SOUNDS FROM THE EB


# 2.2.1. Onset Detection of the EB

import IPython.display as ipd
import librosa
import librosa.display
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn import preprocessing
from sklearn import cluster
#import pydub
import soundfile
import pandas as pd
from adjustText import adjust_text

import os
import math
import glob
from pathlib import Path
import urllib
%matplotlib inline


# Load audio with librosa
# extract onset frames, times and samples
x = 'C:/Users/MuwiHH/Desktop/MARTINEZ DATA SET/Guess The Snare/DATA SET/BeatClean.wav'

x, sr = librosa.load(x, duration = 8.0)
onset_frames = librosa.onset.onset_detect(y=x, backtrack = True)
onset_times = librosa.frames_to_time(onset_frames)
onset_samples = librosa.frames_to_samples(onset_frames)
print (len(onset_samples))

# plot onsets upon the waveform
plt.figure(figsize=(14, 4))
librosa.display.waveshow(x, sr=sr)
plt.vlines(onset_times, -0.8, 0.79, color='r', alpha=0.8)
plt.ylabel('Amplitude (scaled)')
plt.show()
#plt.savefig('on_sets.png', bbox_inches = 'tight')

# listen to extracted onset events
clicks = librosa.clicks(frames=onset_frames, sr=sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)

# 2.2.2. Feature Extraction of the EB

def extract_features(x, fs):
    sf = librosa.feature.spectral_flatness(y=x)[0,0]
    sc = librosa.feature.spectral_centroid(y=x)[0,0]
    return [sf, sc]

# run extract_features across the onset samples
fs = 22050
frame_sz = fs*0.030 
features = np.array([extract_features(x[i:int(i+frame_sz)], fs=fs) for i in onset_samples])

# Normalization of Data, scale from -1 to 1

min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(features)

# 2.2.3. Clustering with K-Means (KM)

# develop a model to find 3 clusters

model = cluster.KMeans(n_clusters=3) 
labels = model.fit_predict(features_scaled) # fit predict 
#print(labels)

plt.figure(figsize=(6, 6))
plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b', marker=".")
plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r', marker=".")
plt.scatter(features_scaled[labels==2,0], features_scaled[labels==2,1], c='y', marker=".")

texts = [plt.text(features_scaled[i,0], features_scaled[i,1], f'{i}') for i in range(features_scaled.shape[0])]
adjust_text(texts, arrowprops={'arrowstyle':'-', 'color':'green'})

plt.xlabel('Spectral Flatness (scaled)')
plt.ylabel('Spectral Centroid (scaled)')
plt.legend(('Cluster 1', 'Cluster 2', 'Cluster 3'))
#plt.savefig('k-means-sf_sc.png', bbox_inches = 'tight')
plt.show()


# listen to each cluster
clicks1 = librosa.clicks(frames=onset_frames[labels==0], sr=sr, length=len(x))
ipd.Audio(x + clicks1, rate=sr)

clicks2 = librosa.clicks(frames=onset_frames[labels==1], sr=sr, length=len(x))
ipd.Audio(x + clicks2, rate=sr)

clicks3 = librosa.clicks(frames=onset_frames[labels==2], sr=sr, length=len(x))
ipd.Audio(x + clicks3, rate=sr)
#print(onset_frames[labels==2])


# 2.2.4. Extract and concatenate snare samples
# Extracting the indexes from each Cluster from the K Means

#snares

# 1) Print labels to visualize information
print(labels == 0)

# 2) Extract the indexes of the values that belong to the cluster in which the snare sound is. 
# (in this case is cluster 2) and send them to the empty list. This would have only the extracted snare sounds.

hh = []
hh.append((np.where(labels == 0)[0]).tolist())
# len(y[0])

#kicks
kicks = []
kicks.append((np.where(labels == 1)[0]).tolist())
# len(y2[0])

# snares
y3 = []
y3.append((np.where(labels == 2)[0]).tolist())
# len(y3[0])

# access the snares from the signal through the extracted indexes and convert them from frames to samples and times
snare_frames = onset_frames[y3]
snare_samples = onset_samples[y3]
snare_times = onset_times[y3]

# kick_frames = onset_frames[y2]
# kick_samples = onset_samples[y2]
# kick_times = onset_times[y2]

# hh_frames = onset_frames[y3]
# hh_samples = onset_samples[y3]
# hh_times = onset_times[y3]

# len(snare_frames[0])
onset_times_sn = librosa.frames_to_time(onset_frames[labels == 2])

# plot snare onsets upon the waveform

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.vlines(onset_times, -0.8, 0.79, color='r', alpha=0.8)
plt.vlines(onset_times_sn, -0.8, 0.79, color='g', alpha=0.8)
#plt.savefig('on_sets_sn.png')
plt.show()

# CONCATENATING

# # kick

# def concatenate_kick(x, kick_samples, pad_duration=0.500):
#     silence = np.zeros(int(pad_duration*sr)) # silence
#     frame_sz = int(min(np.diff(kick_samples)))   # every segment has uniform frame size
#     return np.concatenate([
#         np.concatenate([x[i:i+frame_sz], silence]) # pad segment with silence
#         for i in kick_samples
#     ])

# concatenated_kick = concatenate_kick(x, kick_samples[0])
# print(concatenated_kick)
# ipd.Audio(concatenated_kick, rate=sr)

# # hi hat

# def concatenate_hh(x, hh_samples, pad_duration=0.500):
#     silence = np.zeros(int(pad_duration*sr)) # silence
#     frame_sz = min(np.diff(hh_samples))   # every segment has uniform frame size
#     return np.concatenate([
#         np.concatenate([x[i:i+frame_sz], silence]) # pad segment with silence
#         for i in hh_samples
#     ])


# concatenated_hh = concatenate_hh(x, hh_samples[0])
# print(concatenated_hh)
# ipd.Audio(concatenated_hh, rate=sr)

# snare

def concatenate_snare(x, snare_samples, pad_duration=0.900):
    silence = np.zeros(int(pad_duration*sr)) # silence
    frame_sz = min(np.diff(snare_samples))   # every segment has uniform frame size
    return np.concatenate([
        np.concatenate([x[i:i+frame_sz], silence]) # pad segment with silence
        for i in snare_samples
    ])

concatenated_snare = concatenate_snare(x, snare_samples[0])
print(len(concatenated_snare))
ipd.Audio(concatenated_snare, rate=sr)

# snare_short

def concatenate_short_snare(x, snare_samples, pad_duration=0.900):
    silence = np.zeros(int(pad_duration*sr)) # silence
    frame_sz = int(min(np.diff(snare_samples))*0.45)   # every segment has uniform frame size
    return np.concatenate([
        np.concatenate([x[i:i+frame_sz], silence]) # pad segment with silence
        for i in snare_samples
    ])

concatenated_snare_short = concatenate_short_snare(x, snare_samples[0])
print(len(concatenated_snare_short))
ipd.Audio(concatenated_snare_short, rate=sr)

# Accuracy of the samples present on the concatenated_snare audio vs concatenated_snare_short
# onset detect of the extracted snares (ES)

onset_frames_sn = librosa.onset.onset_detect(y=concatenated_snare, backtrack = True)
onset_times_sn = librosa.frames_to_time(onset_frames_sn)
onset_samples_sn = librosa.frames_to_samples(onset_frames_sn)
# print(len(onset_times_sn))

onset_frames_sn_short = librosa.onset.onset_detect(y=concatenated_snare_short, backtrack = True)
onset_times_sn_short = librosa.frames_to_time(onset_frames_sn_short)
onset_samples_sn_short = librosa.frames_to_samples(onset_frames_sn_short)
# print(len(onset_times_sn_short))

# plot snare onsets upon the waveform
line1 = [0,1,4,5,8, 9, 10, 13, 14]
line2 = [2,3,6,7,11,12,15]
line3 = [0,1,2,3,4,5,6,7,8]
line4 = [9]

plt.figure(figsize=(14, 13))
plt.subplot (2, 1, 1)
plt.title('a. First rendered audio', size='xx-large')
librosa.display.waveshow(concatenated_snare, sr=sr)
plt.vlines(onset_times_sn, -0.8, 0.79, color='r', alpha=0.8)
plt.vlines(onset_times_sn[[line1]], -0.8, 0.79, color='g', alpha=0.8)
plt.vlines(onset_times_sn[[line2]], -0.8, 0.79, color='r', alpha=0.8)

plt.subplot (2, 1, 2)
plt.title('b. Second rendered audio', size='xx-large')
librosa.display.waveshow(concatenated_snare_short, sr=sr)
plt.vlines(onset_times_sn_short[[line3]], -0.8, 0.79, color='g', alpha=0.8)
plt.vlines(onset_times_sn_short[[line4]], -0.8, 0.79, color='r', alpha=0.8)
#plt.savefig('on_sets_snare_short vs snare.png', bbox_inches = 'tight')
plt.show()

# plot snare onsets upon the waveform
line1 = [0,1,2,4,5,6,7]
line2 = [3]
line3 = [8]
line4 = [9]


plt.figure(figsize=(14, 5))

librosa.display.waveshow(concatenated_snare_short, sr=sr)
#plt.vlines(onset_times_sn[[0,1,2,4,5,6,7]], -0.8, 0.79, color='r', alpha=0.8)
plt.vlines(onset_times_sn_short[[line1]], -0.8, 0.79, color='g', alpha=0.8)
plt.vlines(onset_times_sn_short[[line2]], -0.8, 0.79, color='y', alpha=0.8)
plt.vlines(onset_times_sn_short[[line2]], -0.8, 0.79, color='y', alpha=0.8)
plt.vlines(onset_times_sn_short[[line3]], -0.8, 0.79, color='b', alpha=0.8)
plt.vlines(onset_times_sn_short[[line4]], -0.8, 0.79, color='r', alpha=0.8)
#plt.legend([line1, line2, line3, line4], ['label1', 'label2', 'label3', 'label4'])
#plt.savefig('on_sets_snare_short.png', bbox_inches = 'tight')
plt.show()

# create target snares for next step

target_snare_1 = concatenated_snare_short[180000:181600] # Snare02.wav
target_snare_2 = concatenated_snare_short[38:3000] # Snare05.wav
target_snare_3 = concatenated_snare_short[67000:70000] # Snare33.wav

plt.figure(figsize=(14, 14))
plt.subplot(3,1,1)
librosa.display.waveshow(target_snare_1, sr=sr)
plt.subplot(3,1,2)
librosa.display.waveshow(target_snare_2, sr=sr)
plt.subplot(3,1,3)
librosa.display.waveshow(target_snare_3, sr=sr)
plt.show()
