import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ========== STEP 1: LOAD AND PREPROCESS DATA ==========
train_dir = "dataset/train"

train_datagen = ImageDataGenerator(
    rescale=1.0/255, rotation_range=20, zoom_range=0.2,
    width_shift_range=0.2, height_shift_range=0.2,
    horizontal_flip=True, validation_split=0.2  # Optional, remove if no val set
)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
)

# OPTIONAL: If validation data is available
val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary', subset="validation"
)

# ========== STEP 2: BUILD THE MODEL ==========
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

# Create Model
model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ========== STEP 3: TRAIN THE MODEL ==========
epochs = 10
if "val_generator" in locals():
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
else:
    history = model.fit(train_generator, epochs=epochs)

# Save Model
model.save("face_mask_detector.h5")

# ========== STEP 4: LOAD MODEL FOR REAL-TIME DETECTION ==========
model = tf.keras.models.load_model("face_mask_detector.h5")

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess for model
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0  # Normalize
        face_img = np.expand_dims(face_img, axis=0)

        # Predict Mask/No Mask
        prediction = model.predict(face_img)[0][0]
        confidence = abs(prediction - 0.5) * 2  # Convert to range [0,1]
        
        if prediction < 0.5:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        # Draw Rectangle and Label with Confidence Score
        text = f"{label} ({confidence*100:.2f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display Output
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
