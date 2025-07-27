# Image Convolution with Neural Networks

This project explores image classification using convolutional neural networks (CNNs), including both custom-built architectures and transfer learning with pretrained models. The experiments are conducted on the CIFAR-100 dataset and demonstrate the effectiveness of CNN-based feature extraction for image recognition.

## Overview

The repository includes three main components:
- Custom_CNN: A custom CNN built and trained from scratch on CIFAR-100
- FCN: A fully convolutional network designed for classification without dense layers
- VGG16: A fine-tuned version of the pretrained VGG16 model using transfer learning

## Models

### Custom CNN
- Built from scratch using Keras
- Includes convolutional, pooling, dropout, and dense layers
- Trained end-to-end on CIFAR-100 with data augmentation

### Fully Convolutional Network (FCN)
- No dense layers used
- Final classification via global average pooling
- Useful for spatially-aware tasks and more efficient architectures

### VGG16 Transfer Learning
- Pretrained VGG16 model from ImageNet
- Top layers replaced with custom dense layers
- Fine-tuned on CIFAR-100 for 100-class classification

## Dataset

- CIFAR-100: 60,000 color images in 100 classes (500 training and 100 test images per class)
- Automatically downloaded via keras.datasets.cifar100 or integrated in code

## Requirements

- Python 3.7+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- tqdm (optional for progress bars)


