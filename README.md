# BA-NeuralFunk
This repository cointains all code-based files and documentation which contribute to our Bachelor Thesis. The goal is to replicate the given work and extend it with our own ideas.

## Overview
Project structure:

```
BA_NeuralFunk
|
│
└───DeepDrummer
│   │ 
│   └───samples
│   │ 
│   └───save_data
│       │ 
│       └───good
│       │ 
│       └───bad
│
|
└───SpectrogramVAE
│
│
└───data
    │
    └───logdir
    │ 
    └───samples
        │ 
        └───raw_samples

```
## Samples
Download and save in a raw_samples directory (see project structure above)
* [Drumkit](https://s3-ap-northeast-1.amazonaws.com/codepen-dev/drumkit_dataset.zip)
* [NeuroFunk Sample Pack](https://www.ghosthack.de/free_sample_packs/neurofunk-sample-pack/)

## Local anaconda environment setup
* Create new environment with Python 3.7:\
```conda create --name py37 python=3.7```
* Activate new environment:\
```activate py37```
* Navigate to project directory:\
```cd <path>/SpectrogramVAE```
* Install requirements:\
```pip install -r requirements.txt```

## Preprocessing
```python preprocessing.py --data_dir <datadir> --target_dir <targetdir>```
## Training
```python train.py --logdir <logdir>```
## Encode audio
* Single file:\
```python encode_and_reconstruct.py --audio_file <filename>```
* Full dataset:\
```python encode_and_reconstruct.py -logdir <logdir> --encode_full true```
## Generating samples
* Sampling from latent space:\
```python generate_experiments.py --audio_dir <path> --random True```
* Combine multiple samples either random or from one or two specific categories:\
```python generate_experiments.py --audio_dir <path> --combine True```\
```python generate_experiments.py --audio_dir <path> --combine True --cat1 0 --cat2 1```
* Audio reconstruction:\
```python generate_experiments.py --audio_dir <path> --reconstruct True```




### Given Work
* [Neural Funk Article](https://towardsdatascience.com/neuralfunk-combining-deep-learning-with-sound-design-91935759d628) 
* [Neural Funk GitHub Repository (Given Project)](https://github.com/maxfrenzel/SpectrogramVAE) 
* [WaveNet: A generative model for raw audio](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio)
* [Tensorflow Wavenet implementation](https://github.com/ibab/tensorflow-wavenet)
* [Audio Samples](https://www.dropbox.com/s/vo5s1iq5eqyxxcm/Generated%20Samples.zip?dl=0)

### Different Approaches / Models / Related Work
* [Performance RNN](https://magenta.tensorflow.org/performance-rnn)
* [NSynth: Neural Audio Synthesis](https://magenta.tensorflow.org/nsynth)
* [Audio classification from spectrograms](https://gist.github.com/naotokui/a2b331dd206b13a70800e862cfe7da3c)
* [The Variational Autoencoder as a Two-Player Game — Part I](https://towardsdatascience.com/the-variational-autoencoder-as-a-two-player-game-part-i-4c3737f0987b)
* [The Variational Autoencoder as a Two-Player Game — Part II](https://towardsdatascience.com/the-variational-autoencoder-as-a-two-player-game-part-ii-b80d48512f46)
* [The Variational Autoencoder as a Two-Player Game — Part III](https://towardsdatascience.com/the-variational-autoencoder-as-a-two-player-game-part-iii-d8d56c301600)
* [Improved Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/pdf/1606.04934.pdf)
* [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
