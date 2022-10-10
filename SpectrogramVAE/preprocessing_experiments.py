# for debugging only -> https://zohaib.me/debugging-in-google-collab-notebook/
# to set a breakpoint use the following method -> ipdb.set_trace(context=6)
# !pip install -Uqq ipdb
# import ipdb

# %% 
# Imports
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
# get_melspec
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
# Generate melspec from audio-file
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

# %%
drum_dirs = [r for r in sorted(glob('../../data/Samples/drumkit_dataset/*'))]
NB_CLASS = len(drum_dirs)
print(drum_dirs)

# %%
drumkit_dirs = glob("../../data/Samples/drumkit_dataset/*")
ghosthack_dirs = glob("../../data/Samples/Ghosthack_Neurofunk_FreePack/*")

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
                audio_files.append('%s/%s' % (dirName,fname))

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
def get_class_id(filename):
    categories = [
        ['kick'],
        ['snare', 'snr'],
        ['hat', 'hh', 'open', 'closed'],
        ['tom'],
        ['clap'],
        ['rim'],
        ['bass'],
        ['drum'],
        ['fx','riser'],
        ['pad'],
        ['reece']
    ]
    class_id = 0
    for j, cat in enumerate(categories):
        cond = any([s in filename.lower() for s in cat])
        if cond:
            class_id = j+1
            break

    return class_id

# %%
from shutil import copyfile

# Build dataset for Wavenet training
sample_length_sec = 3.0
num_samples_dataset = int(sample_length_sec * SAMPLING_RATE)

dataset_dir = '/root/data/Samples/samples_spec_dataset/'

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
        # class_id = get_class_id(filename)
        # dataset_filename = dataset_dir + str(class_id) + ' - ' + filename + '.wav'
        # dataset_filename_spec = dataset_dir + str(class_id) + ' - ' + filename + '.npy'

        # Write to file
        # outdated: librosa.output.write_wav(dataset_filename, y_tmp, sr, norm=True)
        #script_dir = os.path.dirname(__file__)
        #abs_file_path = os.path.join(script_dir, dataset_filename)

        sf.write(dataset_filename, y_tmp, samplerate=sr) #, norm=True -> not available in SoundFile
        np.save(dataset_filename_spec,spec)

        counter += 1
    
    except:
        pass
# %%
emb = np.load('../SpectrogramVAE-master/logdir/embeddings-9999/0 - Roland TR-909 BD Selection-KK-Di909BD 62.npy')
emb
# %%
emb = np.load('../SpectrogramVAE-master/logdir/embeddings-9999/1 - Roland TR-909 BD Selection-KK-Di909BD 62.npy')
emb
# %%
spec = np.load('../../data/Samples/samples_spec_dataset/0 - Boss DR-202-202snr46.wav.npy')
print(spec.shape)
librosa.display.specshow(spec, sr=SAMPLING_RATE, y_axis='mel', x_axis='time', hop_length=HOP_LENGTH)
# %%
emb_dir = '../SpectrogramVAE-master/logdir/embeddings-9999'

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
    ['snare', 'snr'],
    ['hat', 'hh'],
    ['tom'],
    ['clap'],
    ['rim'],
    ['bass'],
    ['drum'],
    ['fx','riser'],
    ['pad'],
    ['reece']
]


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

# %%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(emb_mat)

#%%
import seaborn as sns
fig1 = plt.figure(figsize=(12, 12))
fig1.add_subplot(111)
# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=emb_cat_names, s=50, marker=".")
# %%
fig2 = plt.figure(figsize=(12, 12))
fig2.add_subplot(111)
# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=emb_cat_names, s=2, marker="x")
# %%
import shlex, subprocess

# Generate a bunch of random files using the generate.py script
for k in range(0,20):
    
    # How many files to combine?
    num_in = np.random.randint(2,4)
    
    #Pick random files
    files_in = []
    for j in range(num_in):
        files_in.append(random.choice(audio_files))
        
    dir_arg = ''
    for file in files_in:
        dir_arg += f'"{file}" '
        
    command_line = f"python generate.py --logdir='../SpectrogramVAE-master/logdir' --file_in {dir_arg} --file_out random{k}"
    
    args = shlex.split(command_line)        
    subprocess.call(args)

# %%
import shlex, subprocess

# Generate a bunch of random files, based on latent space sampling, using the generate.py script
for k in range(20):
        
    command_line = f"python generate.py --logdir='../SpectrogramVAE-master/logdir' --file_in  --file_out sampled{k}"
    
    args = shlex.split(command_line)        
    subprocess.call(args)

# %% Test
# Einzelne samples dekodieren und rekonstruieren
import shlex, subprocess

for k in range(20):
    file = random.choice(audio_files)
    file_formatted = "'" + file + "'"
    command_line = f"python ../SpectrogramVAE-master/encode_and_reconstruct.py --logdir='../SpectrogramVAE-master/logdir' --audio_file {file_formatted}"
    args = shlex.split(command_line)        
    subprocess.call(args)
# %%
print(audio_files[0])
# %%
# Find similar generated samples based on an existing sample
import shlex, subprocess

for k in range(1):
    # file = random.choice(audio_files)
    file = audio_files[k]
    file_formatted = "'" + file + "'"
    command_line = f"python ../SpectrogramVAE-master/find_similar.py --logdir ../SpectrogramVAE-master/logdir --target {file_formatted} --sample_dirs ../../data/Samples/generated"
    args = shlex.split(command_line)        
    subprocess.call(args)

# %%
# Encode every audio sample seperately
import shlex, subprocess

counter = 0
for file in audio_files:
    counter += 1
    print(f"Current file nr: {counter}")
    file_formatted = "'" + file + "'"
    command_line = f"python ../SpectrogramVAE-master/encode_and_reconstruct.py --logdir ../../data/Samples/logdir --audio_file {file_formatted}"
    args = shlex.split(command_line)        
    subprocess.call(args)
    

# %%
