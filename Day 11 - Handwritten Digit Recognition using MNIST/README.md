# Day 11: Handwritten Digit Recognition using MNIST

## Overview
A Convolutional Neural Network (CNN) that recognizes handwritten digits (0-9) using the MNIST dataset.

## Features
- CNN architecture with 3 convolutional layers
- Dropout regularization to prevent overfitting
- Training visualization (accuracy/loss plots)
- Prediction visualization on test samples

## Requirements
```bash
pip install tensorflow numpy matplotlib
```

## Usage
```bash
python mnist_digit_recognition.py
```

## Model Architecture
- Conv2D (32 filters) → MaxPooling
- Conv2D (64 filters) → MaxPooling
- Conv2D (64 filters)
- Dense (64 units) → Dropout → Output (10 classes)

## Expected Results
- Test Accuracy: ~99%
- Training time: ~2-3 minutes on CPU

## Output Files
- `mnist_model.h5` - Trained model
- `training_history.png` - Training curves
- `predictions.png` - Sample predictions
