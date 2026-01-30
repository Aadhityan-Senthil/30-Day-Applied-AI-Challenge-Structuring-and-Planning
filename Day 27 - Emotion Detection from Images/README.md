# Day 27: Emotion Detection from Images

## Overview
Detect facial emotions using a CNN trained on synthetic face data. Classifies 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Requirements
```bash
pip install numpy matplotlib scikit-learn seaborn
pip install tensorflow  # Optional but recommended
```

## Usage
```bash
python emotion_detection.py
```

## Real Dataset
Download FER2013 from Kaggle:
```
https://www.kaggle.com/msambare/fer2013
```

## Model Architecture
- Conv2D blocks with BatchNorm and Dropout
- 3 convolutional stages (32→64→128 filters)
- Dense layers (256→7 classes)

## Output Files
- `emotion_detection_results.png` - Visualizations
- `emotion_results.json` - Performance metrics

## Emotions Detected
| Emotion | Description |
|---------|-------------|
| Happy | Smiling, positive |
| Sad | Frowning, downcast |
| Angry | Tense, furrowed brow |
| Surprise | Wide eyes, open mouth |
| Fear | Wide eyes, tense |
| Disgust | Wrinkled nose |
| Neutral | Relaxed, no expression |

## Real-time Detection
```python
import cv2
# Use OpenCV to capture webcam and detect faces
# Feed cropped face to the model for emotion prediction
```
