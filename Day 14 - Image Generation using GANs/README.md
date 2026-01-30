# Day 14: Image Generation using GANs

## Overview
Implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate handwritten digit images from random noise.

## How GANs Work
1. **Generator**: Creates fake images from random noise
2. **Discriminator**: Distinguishes real images from fake ones
3. **Adversarial Training**: Both networks improve through competition

## Architecture

### Generator
- Dense → Reshape to 7×7×256
- ConvTranspose → 14×14×128
- ConvTranspose → 28×28×64
- ConvTranspose → 28×28×1 (output)

### Discriminator
- Conv2D 64 → LeakyReLU → Dropout
- Conv2D 128 → LeakyReLU → Dropout
- Flatten → Dense (1)

## Requirements
```bash
pip install tensorflow numpy matplotlib
```

## Usage
```bash
python dcgan_image_generation.py
```

## Hyperparameters
- Latent dimension: 100
- Batch size: 128
- Learning rate: 0.0002
- Beta1: 0.5
- Epochs: 50

## Output Files
- `generator_model.h5` - Trained generator
- `final_generated_images.png` - Grid of generated digits
- `training_progress.png` - Loss curves and sample evolution

## Generate New Images
```python
generator = keras.models.load_model('generator_model.h5')
noise = tf.random.normal([1, 100])
generated_image = generator(noise)
```

## References
- Radford et al., "Unsupervised Representation Learning with DCGANs" (2015)
