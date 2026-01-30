# Day 08: Object Detection using OpenCV

## Overview
Real-time object detection using two approaches: Haar Cascade for face detection and YOLOv8 for general object detection. Switch between methods with keyboard shortcuts.

## Features
- Haar Cascade face detection
- YOLOv8 object detection (80 classes)
- Real-time webcam processing
- Keyboard controls to switch modes
- Bounding box visualization

## Requirements
```bash
pip install opencv-python numpy ultralytics
```

## Usage
```bash
python object_detection.py
```

## Controls
| Key | Action |
|-----|--------|
| `h` | Switch to Haar Cascade (face detection) |
| `y` | Switch to YOLOv8 (object detection) |
| `q` | Quit application |

## Detection Methods

### Haar Cascade
- Fast and lightweight
- Good for face detection
- Uses pre-trained XML classifiers
- Works well in controlled lighting

### YOLOv8
- State-of-the-art object detection
- 80 COCO classes (person, car, dog, etc.)
- Real-time performance
- More accurate but requires more compute

## Required Files
- `haarcascade_frontalface_default.xml` - Haar cascade classifier
- `yolov8n.pt` - YOLOv8 nano model (auto-downloads)

## COCO Classes (YOLOv8)
Detects 80 objects including:
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, and many more.

## Performance Tips
- Use good lighting for better detection
- YOLOv8 nano (n) is fastest
- Larger YOLO models (s, m, l) are more accurate
- Haar works best for frontal faces

## Key Learnings
- Haar Cascades use sliding window approach
- YOLO detects objects in single forward pass
- Real-time processing requires optimization
- Different methods suit different use cases
