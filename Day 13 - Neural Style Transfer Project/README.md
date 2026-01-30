# Day 13: Neural Style Transfer

## Overview
Apply the artistic style of one image to the content of another using deep learning. This implementation uses the VGG19 network to extract style and content features.

## How It Works
1. **Content Extraction**: Uses deep layers (block5_conv2) to capture high-level content
2. **Style Extraction**: Uses multiple layers to capture textures via Gram matrices
3. **Optimization**: Iteratively modifies a generated image to minimize combined loss

## Features
- VGG19-based feature extraction
- Gram matrix for style representation
- Configurable style/content weight balance
- Progress visualization

## Requirements
```bash
pip install tensorflow numpy matplotlib pillow
```

## Usage
```bash
python neural_style_transfer.py
```

### With Custom Images
```python
result, _ = style_transfer('your_content.jpg', 'your_style.jpg', iterations=500)
```

## Output Files
- `content_image.jpg` - Sample content image
- `style_image.jpg` - Sample style image
- `stylized_output.jpg` - Final stylized result
- `style_transfer_results.png` - Comparison visualization

## Parameters
- `style_weight`: Controls style influence (default: 1e-2)
- `content_weight`: Controls content preservation (default: 1e4)
- `iterations`: Number of optimization steps (default: 100)

## References
- Gatys et al., "A Neural Algorithm of Artistic Style" (2015)
