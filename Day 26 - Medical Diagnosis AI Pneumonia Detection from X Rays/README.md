# Day 26: Medical Diagnosis AI - Pneumonia Detection from X-Rays

## Overview
Build a CNN to detect pneumonia from chest X-ray images using deep learning.

## ⚠️ Disclaimer
This project is for **educational purposes only**. Do NOT use for actual medical diagnosis.

## Features
- Custom CNN architecture
- Data augmentation
- Transfer learning option (VGG16)
- ROC curve and AUC analysis
- Confusion matrix visualization

## Requirements
```bash
pip install numpy matplotlib scikit-learn seaborn
# For full functionality:
pip install tensorflow
```

## Usage
```bash
python pneumonia_detection.py
```

## Real Dataset
Download from Kaggle:
```
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
```

## Model Architecture
- Conv2D (32) → BatchNorm → MaxPool
- Conv2D (64) → BatchNorm → MaxPool
- Conv2D (128) → BatchNorm → MaxPool
- Dense (128) → Dropout → Output

## Output Files
- `pneumonia_detection_results.png` - Comprehensive visualization
- `sample_xrays.png` - Sample images with predictions
- `diagnosis_results.json` - Performance metrics

## Key Metrics
- **Recall**: Critical for medical diagnosis (catching all cases)
- **Precision**: Avoiding false positives
- **AUC**: Overall model performance
