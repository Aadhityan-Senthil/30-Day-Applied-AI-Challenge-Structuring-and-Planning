"""
Day 11: Handwritten Digit Recognition using MNIST
30-Day AI Challenge

This project builds a neural network to recognize handwritten digits (0-9)
using the classic MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    """Load MNIST dataset and preprocess for neural network."""
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    
    return (x_train, y_train), (x_test, y_test)

def build_cnn_model():
    """Build a Convolutional Neural Network for digit recognition."""
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=5):
    """Train the model and return history."""
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    return history

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("Training history plot saved as 'training_history.png'")

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """Visualize model predictions on test samples."""
    # Get predictions
    predictions = model.predict(x_test[:num_samples], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test[:num_samples], axis=1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
        color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
        ax.set_title(f'Pred: {predicted_labels[i]} | True: {true_labels[i]}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.close()
    print("Predictions visualization saved as 'predictions.png'")

def main():
    print("=" * 50)
    print("Day 11: MNIST Handwritten Digit Recognition")
    print("=" * 50)
    
    # Load data
    print("\n[1] Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build model
    print("\n[2] Building CNN model...")
    model = build_cnn_model()
    model.summary()
    
    # Train model
    print("\n[3] Training model...")
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=5)
    
    # Visualizations
    print("\n[4] Generating visualizations...")
    plot_training_history(history)
    visualize_predictions(model, x_test, y_test)
    
    # Save model
    model.save('mnist_model.h5')
    print("\nModel saved as 'mnist_model.h5'")
    
    print("\n" + "=" * 50)
    print("Day 11 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
