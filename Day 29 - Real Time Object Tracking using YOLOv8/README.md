# Day 29: Real-Time Object Tracking using YOLOv8

## Overview
Detect and track objects in real-time using YOLOv8 and a simple IoU-based tracker.

## Features
- YOLOv8 object detection (80+ COCO classes)
- Simple IoU-based object tracking
- Trajectory visualization
- Webcam and video file support
- Synthetic data demo mode

## Requirements
```bash
pip install numpy matplotlib
pip install opencv-python  # For image/video processing
pip install ultralytics    # For YOLOv8
```

## Usage
```bash
python object_tracking.py
```

## Real-Time Detection
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # Nano (fastest)
# model = YOLO('yolov8s.pt')  # Small
# model = YOLO('yolov8m.pt')  # Medium
# model = YOLO('yolov8l.pt')  # Large (best)

# Webcam
model(source=0, show=True)

# Video file
model('video.mp4', show=True, save=True)

# Image
results = model('image.jpg')
```

## YOLOv8 Models
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n | 6MB | Fastest | Good |
| yolov8s | 22MB | Fast | Better |
| yolov8m | 50MB | Medium | Great |
| yolov8l | 83MB | Slow | Best |

## Output Files
- `object_tracking_results.png` - Tracking visualizations
- `sample_detection.png` - Sample detection image
- `tracking_results.json` - Tracking data

## COCO Classes (80)
Includes: person, car, bicycle, dog, cat, chair, bottle, cell phone, and 72 more!
