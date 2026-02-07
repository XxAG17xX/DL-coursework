# CIFAR-10 Image Classification using CNNs (PyTorch)

This project implements a convolutional neural network (CNN) for image
classification on the CIFAR-10 dataset as part of a Deep Learning module.

The focus of this lab was on model architecture design, regularisation,
and performance under a strict parameter budget.

---

## Problem Overview

CIFAR-10 is a standard benchmark dataset consisting of 32×32 RGB images
across 10 object categories.

The objective was to design and train a CNN that achieves strong validation
accuracy while remaining within a fixed parameter limit.

---

## My Contributions

- Implemented a custom CNN architecture in PyTorch
- Designed multiple convolutional blocks with:
  - Convolution layers
  - Batch Normalisation
  - ReLU activations
  - Max Pooling
- Applied Dropout for regularisation
- Implemented a fully connected classifier head
- Ensured the architecture remained under the allowed parameter budget

---

## Model Architecture

- Multi-stage convolutional network
- Progressive spatial downsampling via pooling
- Batch Normalisation to stabilise training
- Dropout to reduce overfitting
- Final dense layers for classification

A model summary and parameter breakdown are included in the folder.

![Model](https://github.com/XxAG17xX/DL-coursework/blob/main/lab5-cifar10/cifar10_cnn_model_summary.png)

---

## Training Setup

- Optimizer: Adam  
- Loss Function: Cross-Entropy Loss  
- Training with validation monitoring  
- Regularisation via Dropout  

---

## Results

- Achieved approximately **82–83% accuracy** on hidden evaluation data
- Successfully passed all automated evaluation tests
- Demonstrated clear performance improvement over a fully-connected baseline

![TrainignResults](https://github.com/XxAG17xX/DL-coursework/blob/main/lab5-cifar10/cifar10_training_curves.png)

![testDataResults](https://github.com/XxAG17xX/DL-coursework/blob/main/lab5-cifar10/cifar10_server_tests_passed.png.png)

---

## Key Concepts Demonstrated

- CNN architecture design
- Regularisation techniques
- Model capacity vs performance trade-offs
- Training and validation workflow in PyTorch

---

## Notes

- Dataset loading and training utilities were partially scaffolded
  as part of the coursework.
- This repository highlights the model implementation and design decisions.
