# Day 12: Edge Detection and Feature Extraction using OpenCV

## Overview
Comprehensive demonstration of image processing techniques including edge detection, corner detection, contour detection, and feature extraction.

## Techniques Implemented

### Edge Detection
- **Canny**: Multi-stage algorithm with hysteresis thresholding
- **Sobel**: Gradient-based edge detection (X and Y directions)
- **Laplacian**: Second derivative-based detection
- **Prewitt**: Discrete differentiation operator

### Feature Extraction
- **Harris Corner Detection**: Classic corner detection algorithm
- **Shi-Tomasi**: Good Features to Track algorithm
- **Contour Detection**: Shape boundary extraction
- **HOG Features**: Histogram of Oriented Gradients

## Requirements
```bash
pip install opencv-python numpy matplotlib
```

## Usage
```bash
python edge_detection.py
```

## Output Files
- `sample_image.png` - Generated test image with shapes
- `feature_extraction_results.png` - Comparison of all techniques

## Applications
- Object detection preprocessing
- Image segmentation
- Feature matching
- Computer vision pipelines
