# DL-coursework
# Audio Scene Classification using CNNs (PyTorch)

This project implements a convolutional neural network for multi-class
environmental audio scene classification as part of a Deep Learning module.
The goal was to design a complete audio classification pipeline under strict
interface and evaluation constraints, focusing on data preprocessing,
model architecture, and training stability rather than boilerplate code.

---
## Problem Overview

Given short audio recordings belonging to different environmental scenes
(e.g. street, park, public transport), the task is to classify each sample
into one of 15 scene categories.

Key challenges include:
- Variable-length audio inputs
- High-dimensional time–frequency representations
- Generalisation to unseen evaluation data

---

## My Contributions

While the assignment provided a notebook scaffold and evaluation utilities,
the following components were **fully implemented by me**:

### 1. Audio Dataset & Preprocessing
- Implemented a custom PyTorch `Dataset` class to load audio files and metadata
- Normalised variable-length waveforms using padding and cropping
- Extracted time–frequency features:
  - Mel-spectrograms
  - MFCCs
- Combined Mel and MFCC features into a **2-channel input representation**
- Applied light data augmentation (noise injection) during training only

### 2. Model Architecture
- Designed a custom CNN to operate on 2D time–frequency inputs
- Used stacked convolutional blocks with Batch Normalisation and ReLU
- Applied **Adaptive Average Pooling** to remove sensitivity to input length
- Implemented a fully-connected classifier head for final prediction
- Ensured the model stayed within the specified parameter budget

![Model](https://github.com/XxAG17xX/DL-coursework/blob/main/lab6-audio-classification/audio_model_summary.png)

### 3. Training & Evaluation Logic
- Implemented the full training loop in PyTorch
- Used cross-entropy loss for multi-class classification
- Optimised using Adam with a tuned learning rate
- Integrated learning-rate scheduling based on validation performance
- Saved the best-performing model checkpoint during training

![ModelCurves](https://github.com/XxAG17xX/DL-coursework/blob/main/lab6-audio-classification/audio_training_log.png)

---

## Training Setup

- Optimizer: Adam  
- Loss Function: Cross-Entropy Loss  
- Learning Rate: Tuned experimentally  
- Scheduler: Reduce-on-plateau strategy  
- Epochs: Fixed-length training with validation monitoring  

---

## Results

- Successfully passed all automated evaluation and interface tests
- Achieved strong validation accuracy on held-out evaluation data
- Demonstrated stable training behaviour with consistent convergence
  
![Results](https://github.com/XxAG17xX/DL-coursework/blob/main/lab6-audio-classification/audio_server_test_results.png)

Screenshots of training curves, model summaries, and server-side evaluation
results are included in the folder.

---

## Key Concepts Demonstrated

- Audio feature engineering (Mel-spectrograms, MFCCs)
- CNN design for time–frequency data
- Handling variable-length inputs
- Training stability and evaluation under constraints
- Working within a fixed API and automated test environment

---

## Notes

- The original coursework notebook and grading utilities are not included
  in this repository.
- This version isolates the core components I implemented for clarity
  and portfolio presentation.
