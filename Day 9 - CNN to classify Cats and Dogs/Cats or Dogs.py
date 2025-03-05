import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

# Data paths
train_dir = "dataset/train/"
test_dir = "dataset/test/"

# Image Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting training data into train & validation
)

# Training Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

# Validation Data
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Load Pretrained MobileNetV2
base_model = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freezing the base model

# Building the Model
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),  # Prevent Overfitting
    keras.layers.Dense(1, activation="sigmoid")  # Binary Classification
])

# Compile the Model
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train the Model
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=20,  # Increased Epochs for Better Learning
                    verbose=1)

# Save the Model
os.makedirs("model", exist_ok=True)
model.save("model/cat_dog_classifier.h5")

# Plot Training History
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Model Evaluation
train_loss, train_acc = model.evaluate(train_generator)
val_loss, val_acc = model.evaluate(val_generator)
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Function to Predict New Images
def predict_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

# Example Prediction
sample_image = "dataset/test/sample.jpg"
print(f"Prediction: {predict_image(sample_image)}")
