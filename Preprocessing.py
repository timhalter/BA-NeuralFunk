# -*- coding: utf-8 -*-
"""Preprocessing_replica.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_jibkT85NZgGHLh4eSQwXLrMYEBVcHxM
"""

# for debugging only -> https://zohaib.me/debugging-in-google-collab-notebook/
# to set a breakpoint use the following method -> ipdb.set_trace(context=6)
# !pip install -Uqq ipdb
# import ipdb

# %%

# Commented out IPython magic to ensure Python compatibility.
import librosa
import librosa.display
from scipy import signal
import numpy as np
from glob import glob
import os
import sys
import time
import random
from random import shuffle
from tqdm import tqdm
from tqdm import tnrange, tqdm_notebook, notebook
import pickle as pkl
import joblib
import matplotlib.pyplot as plt
# %matplotlib inline

N_FFT = 1024
HOP_LENGTH = 256 
SAMPLING_RATE = 16000
MELSPEC_BANDS = 128

sample_secs = 2
num_samples = int(sample_secs * SAMPLING_RATE)

# %%

# Function to read in an audio file and return a mel spectrogram
def get_melspec(filepath_or_audio, hop_length=HOP_LENGTH, n_mels=MELSPEC_BANDS, n_samples=None, sample_secs=None):

    y_tmp = np.zeros(n_samples)
    
    # Load a little more than necessary as a buffer
    load_duration = None if sample_secs == None else 1.1 * sample_secs
    
    # Load audio file or take given input
    if type(filepath_or_audio)==str:
        y, sr = librosa.core.load(filepath_or_audio, sr = SAMPLING_RATE, mono=True, duration=load_duration)
    else:
        y = filepath_or_audio
        sr = SAMPLING_RATE
    
    # Truncate or pad
    if n_samples:
        #ipdb.set_trace(context=6)
        if len(y) >= n_samples:
            y_tmp = y[:n_samples]
            lentgh_ratio = 1.0
        else:
            y_tmp[:len(y)] = y
            lentgh_ratio = len(y)/n_samples
        
    else:
        # ipdb.set_trace(context=6)
        y_tmp = y
        length_ratio = 1.0        
        
    # sfft -> mel conversion
    melspec = librosa.feature.melspectrogram(y=y_tmp, sr=sr,
                n_fft=N_FFT, hop_length=hop_length, n_mels=n_mels)
    S = librosa.power_to_db(melspec, np.max) 
        
    return S, lentgh_ratio

# %%

test_file = './Fluffy Darabuka.wav'
# test_file = '/Users/Shared/Maschine 2 Library/Samples/Instruments/Wind/Wodden Flute Samples/Wodden Flute A2.wav'
spec, _ = get_melspec(test_file, n_samples=num_samples)
print(spec.shape)

# %%
plt.figure(figsize=(10, 4))
librosa.display.specshow(spec, sr=SAMPLING_RATE, y_axis='mel', x_axis='time', hop_length=HOP_LENGTH)
plt.colorbar(format='%+2.0f dB')
plt.title('Amen Break Mel Spectrogram')
plt.tight_layout()
plt.show

# %%
y, _ = librosa.core.load('./Fluffy Darabuka.wav', sr = SAMPLING_RATE, mono=True, duration=3.0)
plt.figure(figsize=(10, 4))
plt.plot(y)
# plt.colorbar(format='%+2.0f dB')
plt.title('Amen Break')
plt.tight_layout()

# %%
from griffin_lim import griffin_lim
import soundfile as sf

# %%
reconstructed = griffin_lim(spec, 50, 0.1, N_FFT, HOP_LENGTH)
plt.figure(figsize=(10, 4))
plt.plot(reconstructed)
# plt.colorbar(format='%+2.0f dB')
plt.title('Amen Break Reconstructed')
plt.tight_layout()
sf.write('recon.wav', reconstructed/np.max(reconstructed), samplerate=SAMPLING_RATE)

# librosa.output was removed in librosa version 0.8.0 -> new: soundfile.write
# librosa.output.write_wav('recon.wav',reconstructed/np.max(reconstructed),sr=SAMPLING_RATE)

# %%
# Go through sample directories and find all audio files
from glob import glob
drum_dirs = [r for r in sorted(glob('..\Samples\drumkit_dataset/*'))]
NB_CLASS = len(drum_dirs)
print(drum_dirs)

# %%
drumkit_dirs = glob("..\Samples\drumkit_dataset\*")
ghosthack_dirs = glob("..\Samples\Ghosthack_Neurofunk_FreePack\*")

# %%
sample_dirs = drumkit_dirs + ghosthack_dirs
print(sample_dirs)
print(len(sample_dirs))

# %%
audio_files = []

for root_dir in sample_dirs:
    for dirName, subdirList, fileList in os.walk(root_dir, topdown=False):
        for fname in fileList:
            if os.path.splitext(fname)[1] in ['.wav', '.aiff', '.WAV', '.aif', '.AIFF', '.AIF']:
                audio_files.append('%s\%s' % (dirName,fname))

len(audio_files)
print(audio_files[:2])

# %%
from tqdm.notebook import tqdm
# If dataset exists, load, otherwise calcluate
if os.path.isfile('dataset.pkl'):
    print('Loading dataset.')
#     with open('dataset.pkl', 'rb') as handle:
#         dataset = pkl.load(handle)
        
    dataset = joblib.load('dataset.pkl')
    
    print('Dataset loaded.')
        
    filenames = dataset['filenames']
    melspecs = dataset['melspecs']
    actual_lengths = dataset['actual_lengths']
        
else:
    print('Creating dataset.')
    
    # Shuffle files
    shuffle(audio_files)

    # Calculate spectra
    melspecs = []
    filenames = []
    actual_lengths = []

    for filename in audio_files:
        try:
            spec, length = get_melspec(filename, n_samples=num_samples, sample_secs=sample_secs)
            melspecs.append(spec)
            filenames.append(filename)
            actual_lengths.append(length)
        except:
            pass
        
    # Store as pickle file
    dataset = {'filenames' : filenames,
                    'melspecs' : melspecs,
                    'actual_lengths' : actual_lengths}
    joblib.dump(dataset, 'dataset.pkl')
#     with open('dataset.pkl', 'wb') as handle:
#         dataset = {'filenames' : filenames,
#                    'melspecs' : melspecs,
#                    'actual_lengths' : actual_lengths}
#         pkl.dump(dataset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    print('Dataset saved.')

# Convert spectra to array
melspecs = np.array(melspecs)

# %%
filenames_short = filenames[0:500]
melspecs_short = melspecs[0:500]
actual_lengths_short = actual_lengths[0:500]

dataset_short = {'filenames' : filenames_short,
                   'melspecs' : melspecs_short,
                   'actual_lengths' : actual_lengths_short}

joblib.dump(dataset_short, 'dataset_small.pkl')
# %%
librosa.display.specshow(melspecs[-6], sr=SAMPLING_RATE, y_axis='mel', x_axis='time', hop_length=HOP_LENGTH)
librosa.display.specshow(melspecs[1], sr=SAMPLING_RATE, y_axis='mel', x_axis='time', hop_length=HOP_LENGTH)
melspecs.shape

# %%
from shutil import copyfile

# Build dataset for Wavenet training
sample_length_sec = 3.0
num_samples_dataset = int(sample_length_sec * SAMPLING_RATE)

dataset_dir = 'C:/Users/tiimh/Documents/BA_NerualFunk/samples_spec_dataset/'

# Shuffle files
shuffle(audio_files)

counter = 0
# %%
for filename in audio_files:
    try: 
        # Load audio file
        y, sr = librosa.core.load(filename, sr = SAMPLING_RATE, mono=True, duration=sample_length_sec)

        y_tmp = np.zeros(num_samples_dataset)

        # Truncate or pad
        if len(y) >= num_samples_dataset:
            y_tmp = y[:num_samples_dataset]
        else:
            y_tmp[:len(y)] = y

        # Calculate spectrum
        spec, _ = get_melspec(y_tmp, n_samples=num_samples_dataset)

        filename = os.path.splitext(os.path.split(filename)[1])[0]

        dataset_filename = dataset_dir + str(counter) + ' - ' + filename + '.wav'
        dataset_filename_spec = dataset_dir + str(counter) + ' - ' + filename + '.npy'

        # Write to file
        # outdated: librosa.output.write_wav(dataset_filename, y_tmp, sr, norm=True)
        sf.write(dataset_filename, y_tmp, samplerate=sr) #, norm=True -> not available in SoundFile
        np.save(dataset_filename_spec,spec)

        counter += 1
    
    except:
        pass
# %%
emb = np.load('C:/Users/tiimh/Documents/BA_NerualFunk/samples_spec_dataset/30 - Alesis DM5-DM5Kick08.npy')
emb
# %%
spec = np.load('C:/Users/tiimh/Documents/BA_NerualFunk/samples_spec_dataset/30 - Alesis DM5-DM5Kick08.npy')
print(spec.shape)
librosa.display.specshow(spec, sr=SAMPLING_RATE, y_axis='mel', x_axis='time', hop_length=HOP_LENGTH)
# %%
emb_dir = 'C:/Users/tiimh/Documents/BA_NerualFunk/samples_spec_dataset'

emb_files = []
for dirName, subdirList, fileList in os.walk(emb_dir, topdown=False):
        for fname in fileList:
            if os.path.splitext(fname)[1] in ['.npy']:
                emb_files.append('%s/%s' % (dirName,fname))
len(emb_files)
# %%
emb_mat = np.zeros((len(emb_files), 64))
emb_cats = []
emb_cat_names = []
categories = [
    ['kick'],
    ['snare'],
    ['hat'],
    ['clap'],
    ['choir'],
    ['sfx'],
    ['piano','rhodes'],
    ['sax']
]

# for k, file in enumerate(emb_files):
#     emb_mat[k] = np.load(file)
#     for j, cat in enumerate(categories):
#         cond = any([s in file.lower() for s in cat])
#         if cond:
#             emb_cats.append(j+1)
#             emb_cat_names.append(cat[0])
#             break
#         else:
#             emb_cats.append(0)
#             emb_cat_names.append('None')

# counter = 0
# for k, file in enumerate(emb_files):
#     for j, cat in enumerate(categories):
#         cond = any([s in file.lower() for s in cat])
#         if cond:
#             emb_mat[counter] = np.load(file)
#             emb_cats.append(j+1)
#             emb_cat_names.append(cat[0])
#             counter+=1
#             break
# emb_mat = emb_mat[:counter]

counter = 0
for k, file in enumerate(emb_files):
    cat_found = False
    for j, cat in enumerate(categories):
        cond = any([s in file.lower() for s in cat])
        if cond:
            emb_mat[counter] = np.load(file)
            emb_cats.append(j+1)
            emb_cat_names.append(cat[0])
            counter+=1
            cat_found = True
            break
    if not cat_found: 
        emb_mat[counter] = np.load(file)
        emb_cats.append(0)
        emb_cat_names.append('Other')
        counter+=1
emb_mat = emb_mat[:counter]
    
    
#     if "kick" in file.lower():
#         emb_cats.append(1)
#         emb_cat_names.append('kick')
#     elif "snare" in file.lower():
#         emb_cats.append(2)
#         emb_cat_names.append('snare')
#     elif "hat" in file.lower():
#         emb_cats.append(3)
#         emb_cat_names.append('hat')
#     elif "clap" in file.lower():
#         emb_cats.append(4)
#         emb_cat_names.append('clap')
#     elif "choir" in file.lower():
#         emb_cats.append(5)
#         emb_cat_names.append('choir')
#     elif "sfx" in file.lower():
#         emb_cats.append(6)
#         emb_cat_names.append('sfx')
#     elif ("rhodes" in file.lower()) or ("piano" in file.lower()):
#         emb_cats.append(7)
#         emb_cat_names.append('piano')
#     elif "sax" in file.lower():
#         emb_cats.append(8)
#         emb_cat_names.append('sax')
#     else:
#         emb_cats.append(0)
# %%