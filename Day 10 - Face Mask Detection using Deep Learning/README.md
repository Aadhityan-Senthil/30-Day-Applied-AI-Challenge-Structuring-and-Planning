# Day 10: Face Mask Detection using Deep Learning

## Overview
Build a real-time face mask detection system using MobileNetV2 transfer learning. Detects faces using Haar Cascade and classifies mask/no-mask with a CNN.

## Features
- Transfer learning with MobileNetV2
- Real-time webcam detection
- Haar Cascade face detection
- Confidence score display
- Color-coded bounding boxes (green=mask, red=no mask)

## Requirements
```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Usage
```bash
python face_mask_detection.py
```

Press `q` to quit the application.

## Dataset Structure
```
dataset/
└── train/
    ├── with_mask/
    │   ├── mask_001.jpg
    │   └── ...
    └── without_mask/
        ├── no_mask_001.jpg
        └── ...
```

Download from: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

## Model Architecture
```
MobileNetV2 (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu)
    ↓
Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

## Training Parameters
- Image size: 224x224
- Batch size: 32
- Epochs: 10
- Optimizer: Adam
- Loss: Binary Crossentropy

## Detection Pipeline
1. Capture frame from webcam
2. Convert to grayscale
3. Detect faces using Haar Cascade
4. For each face:
   - Crop and resize to 224x224
   - Normalize pixel values
   - Predict mask/no-mask
   - Draw bounding box with label

## Output Files
- `face_mask_detector.h5` - Trained model

## Color Coding
- 🟢 **Green**: Mask detected
- 🔴 **Red**: No mask detected

## Performance Tips
- Good lighting improves detection
- Face should be clearly visible
- Multiple faces supported
- Adjust `scaleFactor` and `minNeighbors` for sensitivity

## Key Learnings
- Two-stage detection (face → mask classification)
- Transfer learning enables small dataset training
- Real-time inference requires efficient models
- Haar Cascade provides fast face localization
