"""
Created on Mon Sep 25 08:07:53 2023
@author: Cristhiam Martínez crismartinez@t-online.de
"""

import IPython.display as ipd
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn import preprocessing
from sklearn import cluster
import soundfile
import pandas as pd
from adjustText import adjust_text
import soundfile as sf

import os
import math
import glob
from pathlib import Path
import urllib
#%matplotlib inline

def extract_features(x, fs):
    sf = librosa.feature.spectral_flatness(y=x)[0,0]
    sc = librosa.feature.spectral_centroid(y=x)[0,0]
    return [sf, sc]

def concatenate_short_onsets(x, snare_samples, pad_duration=0.900):
    silence = np.zeros(int(pad_duration*sr)) # silence
    frame_sz = int(min(np.diff(snare_samples))*0.45)   # every segment has uniform frame size
    return np.concatenate([
        np.concatenate([x[i:i+frame_sz], silence]) # pad segment with silence
        for i in snare_samples
    ])

def concatenate_onsets(x, snare_samples, pad_duration=0.900):
    silence = np.zeros(int(pad_duration*sr)) # silence
    frame_sz = min(np.diff(snare_samples))   # every segment has uniform frame size
    return np.concatenate([
        np.concatenate([x[i:i+frame_sz], silence]) # pad segment with silence
        for i in snare_samples
    ])

def write_click_wav(reference_audio, sr):
    for i in range(n_clusters):
        click_track = librosa.clicks (frames=onset_frames[labels == i], sr=sr, length=len (reference_audio))
        # Example audio file path to save
        output_path = f'Cluster{i+1}.wav'
        # Extract audio data from clicks1
        audio_data = reference_audio + click_track
        # Write the audio data to a WAV file
        sf.write (output_path, audio_data, sr)

def extract_onset_information(audio):
    frames = librosa.onset.onset_detect (y=audio, backtrack=True)
    times = librosa.frames_to_time (frames)
    samples = librosa.frames_to_samples (frames)
    return frames, times, samples

def extract_onset_indices(onset_frames, onset_samples, onset_times, indices):
    """access the snares from the signal through the extracted
    indexes and convert them from frames to samples and times"""
    frames = onset_frames[indices]
    samples = onset_samples[indices]
    times = onset_times[indices]
    return frames, samples, times

# Load audio with librosa
original_audio = r'C:\Users\LEGION\PycharmProjects\GuessTheSnare\BeatClean.wav'
original_audio, sr = librosa.load(original_audio, duration = 8.0)

# # extract onset frames, times and samples from reference audio
# onset_frames = librosa.onset.onset_detect(y=original_audio, backtrack = True)
# onset_times = librosa.frames_to_time(onset_frames)
# onset_samples = librosa.frames_to_samples(onset_frames)
# print (len(onset_samples))
#

# extract onset frames, times and samples from reference audio
onset_frames, onset_times, onset_samples = extract_onset_information (audio=original_audio)

# plot onsets upon the waveform
plt.figure(figsize=(14, 4))
librosa.display.waveshow(original_audio, sr=sr)
plt.vlines(onset_times, -0.8, 0.79, color='r', alpha=0.8, label='Onsets')
plt.title('Onsets from reference audio', fontsize='xx-large')
plt.ylabel('Amplitude (scaled)', fontsize='xx-large')
plt.tight_layout()
plt.legend(fontsize='xx-large')
plt.show()
#plt.savefig('on_sets.png', bbox_inches = 'tight')

#%%

# listen to extracted onset events
clicks = librosa.clicks(frames=onset_frames, sr=sr, length=len(original_audio))
ipd.Audio(original_audio + clicks, rate=sr)

# 2.2.2. Feature Extraction of the EB
# run extract_features across the onset samples
fs = 22050
frame_sz = fs*0.030
features = np.array([extract_features(original_audio[i:int(i + frame_sz)], fs=fs) for i in onset_samples])

# Normalization of Data, scale from -1 to 1
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(features)

# 2.2.3. Clustering with K-Means (KM)

# develop a model to find 3 clusters (Kick, hh and snare)
n_clusters = 3
model = cluster.KMeans(n_clusters=n_clusters)
labels = model.fit_predict(features_scaled) # fit predict

plt.figure(figsize=(8, 6))
colors = ['b', 'r', 'y']
labels_text = ['Cluster 1', 'Cluster 2', 'Cluster 3']

for label, color, label_text in zip(range(3), colors, labels_text):
    plt.scatter(features_scaled[labels == label, 0], features_scaled[labels == label, 1], c=color, marker=".", label=label_text)

# Add index labels to each sample
texts = [plt.text(features_scaled[i, 0], features_scaled[i, 1], f'{i}') for i in range(features_scaled.shape[0])]
adjust_text(texts, arrowprops={'arrowstyle': '-', 'color': 'green'})

plt.xlabel('Spectral Flatness (scaled)')
plt.ylabel('Spectral Centroid (scaled)')
plt.title('Clustering from onsets from reference audio')
plt.tight_layout()
plt.legend()
plt.show()

# write wav files for all clusters with click
write_click_wav (reference_audio=original_audio, sr=sr)

# 2.2.4. Extract indices and concatenate snare samples
# Extracting the indexes from each Cluster from the K Means
extracted_kick = [i for i, label in enumerate(labels) if label == 0]
extracted_snare = [i for i, label in enumerate(labels) if label == 1]
extracted_hh = [i for i, label in enumerate(labels) if label == 2]


extracted_sn_frames, extracted_sn_samples, extracted_sn_times = extract_onset_indices(onset_frames,
                                                                                      onset_samples,
                                                                                      onset_times,
                                                                                      extracted_snare)
extracted_kick_frames, extracted_kick_samples, extracted_kick_times = extract_onset_indices(onset_frames,
                                                                                            onset_samples,
                                                                                            onset_times,
                                                                                            extracted_kick)
extracted_hh_frames, extracted_hh_samples, extracted_hh_times = extract_onset_indices(onset_frames,
                                                                                      onset_samples,
                                                                                      onset_times,
                                                                                      extracted_hh)

# plot snare onsets upon the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(original_audio, sr=sr)
# plt.vlines(onset_times, -0.8, 0.79, color='r', alpha=0.8)
plt.vlines(extracted_sn_times, -0.8, 0.79, color='g', alpha=0.8, label='Onset')
plt.ylabel('Amplitude (norm.)')
plt.title('Onsets from extracted snare', fontsize = 'xx-large')
plt.legend(fontsize='xx-large')
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

# Export concatenated_snare as a WAV file
concatenated_snare = concatenate_onsets(original_audio, extracted_sn_samples)
print(concatenated_snare)
sf.write('concatenated_snare.wav', concatenated_snare, sr)
#
# # snare_short
# concatenated_snare_short = concatenate_short_onsets(original_audio, extracted_snare_samples)
# sf.write('concatenated_snare_short.wav', concatenated_snare_short, sr)
#
# # Accuracy of the samples present on the concatenated_snare audio vs concatenated_snare_short
# # onset detect of the extracted snares (ES)
#
# onset_frames_sn = librosa.onset.onset_detect(y=concatenated_snare, backtrack = True)
# onset_times_sn = librosa.frames_to_time(onset_frames_sn)
# onset_samples_sn = librosa.frames_to_samples(onset_frames_sn)
# # print(len(onset_times_sn))
#
# onset_frames_sn_short = librosa.onset.onset_detect(y=concatenated_snare_short, backtrack = True)
# onset_times_sn_short = librosa.frames_to_time(onset_frames_sn_short)
# onset_samples_sn_short = librosa.frames_to_samples(onset_frames_sn_short)
# # print(len(onset_times_sn_short))
#
# # plot snare onsets upon the waveform
# line1 = [0,1,4,5,8, 9, 10, 13, 14]
# line2 = [2,3,6,7,11,12,15]
# line3 = [0,1,2,3,4,5,6,7,8]
# line4 = [9]
#
# plt.figure(figsize=(14, 13))
# plt.subplot (2, 1, 1)
# plt.title('a. First rendered audio', size='xx-large')
# librosa.display.waveshow(concatenated_snare, sr=sr)
# plt.vlines(onset_times_sn, -0.8, 0.79, color='r', alpha=0.8)
# plt.vlines(onset_times_sn[[line1]], -0.8, 0.79, color='g', alpha=0.8)
# plt.vlines(onset_times_sn[[line2]], -0.8, 0.79, color='r', alpha=0.8)
#
# plt.subplot (2, 1, 2)
# plt.title('b. Second rendered audio', size='xx-large')
# librosa.display.waveshow(concatenated_snare_short, sr=sr)
# plt.vlines(onset_times_sn_short[[line3]], -0.8, 0.79, color='g', alpha=0.8)
# plt.vlines(onset_times_sn_short[[line4]], -0.8, 0.79, color='r', alpha=0.8)
# #plt.savefig('on_sets_snare_short vs snare.png', bbox_inches = 'tight')
# plt.show()
#
# # plot snare onsets upon the waveform
# line1 = [0,1,2,4,5,6,7]
# line2 = [3]
# line3 = [8]
# line4 = [9]
#
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(concatenated_snare_short, sr=sr)
# #plt.vlines(onset_times_sn[[0,1,2,4,5,6,7]], -0.8, 0.79, color='r', alpha=0.8)
# plt.vlines(onset_times_sn_short[[line1]], -0.8, 0.79, color='g', alpha=0.8)
# plt.vlines(onset_times_sn_short[[line2]], -0.8, 0.79, color='y', alpha=0.8)
# plt.vlines(onset_times_sn_short[[line2]], -0.8, 0.79, color='y', alpha=0.8)
# plt.vlines(onset_times_sn_short[[line3]], -0.8, 0.79, color='b', alpha=0.8)
# plt.vlines(onset_times_sn_short[[line4]], -0.8, 0.79, color='r', alpha=0.8)
# #plt.legend([line1, line2, line3, line4], ['label1', 'label2', 'label3', 'label4'])
# #plt.savefig('on_sets_snare_short.png', bbox_inches = 'tight')
# plt.show()
#
# # create target snares for next step
# target_snare_1 = concatenated_snare_short[180000:181600] # Snare02.wav
# target_snare_2 = concatenated_snare_short[38:3000] # Snare05.wav
# target_snare_3 = concatenated_snare_short[67000:70000] # Snare33.wav
#
# plt.figure(figsize=(14, 14))
# plt.subplot(3,1,1)
# librosa.display.waveshow(target_snare_1, sr=sr)
# plt.subplot(3,1,2)
# librosa.display.waveshow(target_snare_2, sr=sr)
# plt.subplot(3,1,3)
# librosa.display.waveshow(target_snare_3, sr=sr)
# plt.show()

