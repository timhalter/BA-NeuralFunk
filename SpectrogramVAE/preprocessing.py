import argparse
import os
import pickle as pkl
from glob import glob
from random import shuffle

import joblib
import librosa
import librosa.display
import numpy as np
import soundfile as sf

data_dir = '../data/samples/raw_samples'
target_dir = '../data/samples/dataset/'
N_FFT = 1024
HOP_LENGTH = 256 
SAMPLING_RATE = 16000
MELSPEC_BANDS = 128

sample_secs = 2
num_samples = int(sample_secs * SAMPLING_RATE)

def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--data_dir', type=str, default=data_dir,
                        help='The directory containing the raw audio samples.')
    parser.add_argument('--target_dir', type=str, default=target_dir,
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

def get_audio_dirs(dir):
    return glob(dir + "/*/", recursive=True)

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

def build_dataset(audio_files, target_directory):
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

            dataset_filename = target_directory + str(counter) + ' - ' + filename + '.wav'
            dataset_filename_spec = target_directory + str(counter) + ' - ' + filename + '.npy'

            sf.write(dataset_filename, y_tmp, samplerate=sr) #, norm=True -> not available in SoundFile
            np.save(dataset_filename_spec,spec)

            counter += 1
        
        except:
            pass


def main():
    args = get_arguments()

    print("Loading audio directories...")

    sample_dirs = get_audio_dirs(args.data_dir)

    print("Audio directories loaded!")
    print("Loading audio files...")

    audio_files = get_audio_files(sample_dirs)

    print("Audio files loaded!")
    print("Creating dataset...")
    create_dataset(audio_files)

    print("Dataset created!")
    print("Build dataset...")

    build_dataset(audio_files, args.target_dir)
    
    print("Dataset built! Finished!")

if __name__ == '__main__':
    main()