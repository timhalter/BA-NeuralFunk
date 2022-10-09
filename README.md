# BA-NeuralFunk
This repository cointains all code-based files and documentation which contribute to our Bachelor Thesis. The goal is to replicate the given work and extend it with our own ideas.

## Overview
Project structure:

```
BA_NeuralFunk  
│
└───SpectrogramVAE
│   
└───WaveNet
│
│
data
│
└───samples
│   │
│   └───dataset
│   │   │   sample1.wav
│   │   │   sample1.npy
│   │   │   ...
│   │
│   └───generated
│   │
│   └───raw_samples

```
## Preprocessing
```python preprocessing.py --data_dir <datadir> --target_dir <targetdir>```
## Training
```python train.py --logdir <logdir>```
## Generating samples
* Sampling from latent space:\
```python generate.py --logdir <logdir> --file_out <filename>```
* Sinlge input file:\
```python generate.py --logdir <logdir> --file_in <filename>```
* Multiple input files:\
```python generate.py --logdir <logdir> --file_in <list_of_files> --file_out <filename>```


&emsp; 

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
