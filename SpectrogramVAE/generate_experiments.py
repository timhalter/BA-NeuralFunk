import argparse
import random
import re
import shlex
import subprocess

import numpy as np
import preprocessing

def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--audio_dir', type=str, default="../data/samples/raw_samples",
                    help='The target directory for the generated dataset.')     
    parser.add_argument('--combine', type=bool, default=False,
                        help='Combination mode.')
    parser.add_argument('--random', type=bool, default=False,
                        help='Random sampling mode.')
    parser.add_argument('--reconstruct', type=bool, default=False,
                    help='Reconstruction mode.')    
    parser.add_argument('--cat1', type=int, default=None,
                    help='Instrument 1 which should be combined.')       
    parser.add_argument('--cat2', type=int, default=None,
                    help='Instrument 2 which should be combined.')     
    parser.add_argument('--file_prefix', type=str, default="generated",
                    help='File prefix for generated audio files.')
    parser.add_argument('--outputs', type=int, default=2,
                    help='Number of files which should be generated.')       
    parser.add_argument('--combinations', type=int, default=2,
                    help='Number of audio-files that should be combined.')                                                                                
    return parser.parse_args()

def get_samples_by_category(audio_files):
    
    kick = []
    snare = []  # snare, snr
    hat = []    # hat, hh
    tom = []
    clap = []
    rim = []
    bass = []
    drum = []
    riser = []  # riser, fx
    pad = []
    reece = []
    rest = []   # remaining files

    audio_categories = [kick, snare, hat, tom, clap, rim, bass, drum, riser, pad ,reece]

    for k, file in enumerate(audio_files):
        if re.search('kick', file.lower()):
            audio_categories[0].append(audio_files[k])
        elif re.search('snare', file.lower()) or re.search('snr', file.lower()):
            audio_categories[1].append(audio_files[k])
        elif re.search('hat', file.lower()) or re.search('hh', file.lower()):
            audio_categories[2].append(audio_files[k])
        elif re.search('tom', file.lower()):
            audio_categories[3].append(audio_files[k])
        elif re.search('clap', file.lower()):
            audio_categories[4].append(audio_files[k])
        elif re.search('rim', file.lower()):
            audio_categories[5].append(audio_files[k])
        elif re.search('bass', file.lower()):
            audio_categories[6].append(audio_files[k])
        elif re.search('drum', file.lower()):
            audio_categories[7].append(audio_files[k])
        elif re.search('riser', file.lower()) or re.search('fx', file.lower()):
            audio_categories[8].append(audio_files[k])
        elif re.search('pad', file.lower()):
            audio_categories[9].append(audio_files[k])
        elif re.search('reece', file.lower()):
            audio_categories[10].append(audio_files[k])
        else:
            rest.append(audio_files[k])

    return audio_categories

def generate_combined(audio_files, audio_tobecombined=None, prefix="combination", outputs=2, combinations=2):
    # Generate a bunch of random files using the generate.py script
    print(f"Number of outputs: {outputs}")
    print(f"Number of combinations: {combinations}")
    for k in range(0,outputs):
        
        if audio_tobecombined is not None:
            #Pick random files
            files_in = []
            
            for j in range(combinations):
                if j % 2 == 0:
                    files_in.append(random.choice(audio_files))
                else:
                    files_in.append(random.choice(audio_tobecombined))                
        else:
            #Pick random files
            files_in = []
            for j in range(combinations):
                files_in.append(random.choice(audio_files))
            
        dir_arg = ''
        for file in files_in:
            print(file)
            dir_arg += f'"{file}" '
        
        filename = prefix + str(k)
        command_line = f"python generate.py --logdir='../../data/logdir' --file_in {dir_arg} --file_out {filename}"
        
        args = shlex.split(command_line)        
        subprocess.call(args)

def generate_random(outputs=2):
    # Generate a bunch of random files, based on latent space sampling, using the generate.py script
    for k in range(outputs):            
        command_line = f"python generate.py --logdir='../../data/logdir' --file_in  --file_out sampled{k}"
        args = shlex.split(command_line)        
        subprocess.call(args)

def generate_single_reconstructed(audio_files, outputs=2):
    # Einzelne samples dekodieren und rekonstruieren
    for k in range(outputs):
        file = random.choice(audio_files)
        file_formatted = "'" + file + "'"
        command_line = f"python encode_and_reconstruct.py --logdir='../../data/logdir' --audio_file {file_formatted}"
        args = shlex.split(command_line)        
        subprocess.call(args)

def find_similar_samples(audio_file):
    # Find similar generated samples based on an existing sample
    file_formatted = "'" + audio_file + "'"
    command_line = f"python find_similar.py --logdir ../../data/logdir --target {file_formatted} --sample_dirs ../../data/samples/generated"
    args = shlex.split(command_line)        
    subprocess.call(args)

def main():
    args = get_arguments()

    sample_dirs = preprocessing.get_audio_dirs(args.audio_dir)
    audio_files = preprocessing.get_audio_files(sample_dirs)
    samples_filtered = get_samples_by_category(audio_files)
    
    # Categories by number:
    # 
    # 0: kick
    # 1: snare
    # 2: hat
    # 3: tom
    # 4: clap
    # 5: rim
    # 6: bass
    # 7: drum
    # 8: riser
    # 9: pad
    # 10: reece
    #------------------------

    if args.combine == True:
        # if no categories were passed
        if args.cat1 is None and args.cat2 is None:
            generate_combined(audio_files, prefix=args.file_prefix, outputs=args.outputs, combinations=args.combinations)
        # if only one categorie has been passed
        elif args.cat1 is not None and args.cat2 is None:
            generate_combined(samples_filtered[args.cat1], prefix=args.file_prefix, outputs=args.outputs, combinations=args.combinations)
        # if two categories were passed
        elif args.cat1 is not None and args.cat2 is not None:
            generate_combined(samples_filtered[args.cat1], audio_tobecombined=samples_filtered[args.cat2], prefix=args.file_prefix, outputs=args.outputs, combinations=args.combinations)
    elif args.random == True:
        generate_random(args.outputs)
    elif args.reconstruct == True:
        generate_single_reconstructed(audio_files, args.outputs)

if __name__ == '__main__':
    main()