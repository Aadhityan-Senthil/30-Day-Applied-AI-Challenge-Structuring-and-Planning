# Day 09: CNN to Classify Cats and Dogs

## Overview
Build a Convolutional Neural Network using transfer learning (MobileNetV2) to classify images as cats or dogs with data augmentation.

## Features
- Transfer learning with MobileNetV2
- Image data augmentation
- Binary classification
- Training visualization
- Single image prediction function

## Requirements
```bash
pip install tensorflow numpy matplotlib
```

## Usage
```bash
python cats_dogs_classifier.py
```

## Dataset Structure
```
dataset/
├── train/
│   ├── cats/
│   │   ├── cat.1.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog.1.jpg
│       └── ...
└── test/
    └── sample.jpg
```

Download from: https://www.kaggle.com/c/dogs-vs-cats/data

## Model Architecture
```
MobileNetV2 (frozen base)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu)
    ↓
BatchNormalization
    ↓
Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

## Data Augmentation
- Rotation: ±30°
- Width/Height shift: 20%
- Shear: 20%
- Zoom: 20%
- Horizontal flip

## Training Parameters
- Image size: 150x150
- Batch size: 32
- Epochs: 20
- Optimizer: Adam
- Loss: Binary Crossentropy

## Output Files
- `model/cat_dog_classifier.h5` - Saved model
- Training accuracy/loss plots

## Predict New Images
```python
def predict_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"
```

## Key Learnings
- Transfer learning speeds up training significantly
- Data augmentation prevents overfitting
- Freezing base model preserves learned features
- MobileNetV2 is efficient for mobile/embedded deployment
