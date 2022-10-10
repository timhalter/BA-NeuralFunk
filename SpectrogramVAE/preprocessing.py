import argparse
import librosa
import librosa.display
import numpy as np
from glob import glob
import os
import random
from random import shuffle
import pickle as pkl
import joblib
import matplotlib.pyplot as plt
import shlex, subprocess
import seaborn as sns
from sklearn.manifold import TSNE
from shutil import copyfile
from griffin_lim import griffin_lim
import soundfile as sf

DATA_DIRECTORY = '../../data/samples/raw_samples'
TARGET_DIRECTORY = '../../data/samples/dataset/samples_spec_dataset/'
N_FFT = 1024
HOP_LENGTH = 256 
SAMPLING_RATE = 16000
MELSPEC_BANDS = 128

sample_secs = 2
num_samples = int(sample_secs * SAMPLING_RATE)

def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the raw audio samples.')
    parser.add_argument('--target_dir', type=str, default=TARGET_DIRECTORY,
                        help='The target directory for the generated dataset.')
    return parser.parse_args()

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
        if len(y) >= n_samples:
            y_tmp = y[:n_samples]
            length_ratio = 1.0
        else:
            y_tmp[:len(y)] = y
            length_ratio = len(y)/n_samples
        
    else:
        y_tmp = y
        length_ratio = 1.0        
        
    # sfft -> mel conversion
    melspec = librosa.feature.melspectrogram(y=y_tmp, sr=sr,
                n_fft=N_FFT, hop_length=hop_length, n_mels=n_mels)
    S = librosa.power_to_db(melspec, np.max) 
        
    return S, length_ratio

def get_audio_dirs():
    # drumkit_dirs = glob("../../data/Samples/drumkit_dataset/*")
    # ghosthack_dirs = glob("../../data/Samples/Ghosthack_Neurofunk_FreePack/*")
    # sample_dirs = drumkit_dirs + ghosthack_dirs
    return glob(DATA_DIRECTORY + "/*/", recursive=True)

def get_audio_files(sample_dirs):
    audio_files = []

    for root_dir in sample_dirs:
        for dirName, subdirList, fileList in os.walk(root_dir, topdown=False):
            for fname in fileList:
                if os.path.splitext(fname)[1] in ['.wav', '.aiff', '.WAV', '.aif', '.AIFF', '.AIF']:
                    audio_files.append('%s/%s' % (dirName,fname))
    return audio_files

def create_dataset(audio_files):    
    # If dataset exists, load, otherwise calcluate
    if os.path.isfile('dataset.pkl'):
        print('Loading dataset.')
            
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


    filenames_short = filenames[0:500]
    melspecs_short = melspecs[0:500]
    actual_lengths_short = actual_lengths[0:500]

    dataset_short = {'filenames' : filenames_short,
                    'melspecs' : melspecs_short,
                    'actual_lengths' : actual_lengths_short}

    joblib.dump(dataset_short, 'dataset_small.pkl')

def build_dataset(audio_files):
    # Build dataset for Wavenet training
    sample_length_sec = 3.0
    num_samples_dataset = int(sample_length_sec * SAMPLING_RATE)

    # Shuffle files
    shuffle(audio_files)

    counter = 0

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

            dataset_filename = TARGET_DIRECTORY + str(counter) + ' - ' + filename + '.wav'
            dataset_filename_spec = TARGET_DIRECTORY + str(counter) + ' - ' + filename + '.npy'

            sf.write(dataset_filename, y_tmp, samplerate=sr) #, norm=True -> not available in SoundFile
            np.save(dataset_filename_spec,spec)

            counter += 1
        
        except:
            pass

def emb_similarity_search():
    emb_dir = '../SpectrogramVAE-master/logdir/embeddings-9999'

    emb_files = []
    for dirName, subdirList, fileList in os.walk(emb_dir, topdown=False):
            for fname in fileList:
                if os.path.splitext(fname)[1] in ['.npy']:
                    emb_files.append('%s/%s' % (dirName,fname))
    len(emb_files)

    emb_mat = np.zeros((len(emb_files), 64))
    emb_cats = []
    emb_cat_names = []
    categories = [
        ['kick'],
        ['snare', 'snr'],
        ['hat', 'hh'],
        ['tom'],
        ['clap'],
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
    

    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(emb_mat)


    fig1 = plt.figure(figsize=(12, 12))
    fig1.add_subplot(111)
    # cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    ax = sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=emb_cat_names, s=50, marker=".")

    fig2 = plt.figure(figsize=(12, 12))
    fig2.add_subplot(111)
    # cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    ax = sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=emb_cat_names, s=2, marker="x")

def generate_combined(audio_files):
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

def generate_random():
    # Generate a bunch of random files, based on latent space sampling, using the generate.py script
    for k in range(20):            
        command_line = f"python generate.py --logdir='../SpectrogramVAE-master/logdir' --file_in  --file_out sampled{k}"
        args = shlex.split(command_line)        
        subprocess.call(args)

def generate_single_reconstructed(audio_files):
    # Einzelne samples dekodieren und rekonstruieren
    for k in range(20):
        file = random.choice(audio_files)
        file_formatted = "'" + file + "'"
        command_line = f"python ../SpectrogramVAE-master/encode_and_reconstruct.py --logdir='../SpectrogramVAE-master/logdir' --audio_file {file_formatted}"
        args = shlex.split(command_line)        
        subprocess.call(args)

def find_similar_samples(audio_files):
    # Find similar generated samples based on an existing sample
    for k in range(1):
        # file = random.choice(audio_files)
        file = audio_files[k]
        file_formatted = "'" + file + "'"
        command_line = f"python ../SpectrogramVAE-master/find_similar.py --logdir ../SpectrogramVAE-master/logdir --target {file_formatted} --sample_dirs ../../data/Samples/generated"
        args = shlex.split(command_line)        
        subprocess.call(args)

def main():
    sample_dirs = get_audio_dirs()
    audio_files = get_audio_files(sample_dirs)
    create_dataset(audio_files)
    build_dataset(audio_files)

if __name__ == '__main__':
    main()