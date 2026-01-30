"""
Day 26: Medical Diagnosis AI - Pneumonia Detection from X-Rays
30-Day AI Challenge

Build a CNN to detect pneumonia from chest X-ray images.
Uses synthetic data for demo, can be extended with real dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, 
                                         Dropout, BatchNormalization, GlobalAveragePooling2D)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.applications import VGG16, ResNet50
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def generate_synthetic_xray_data(n_samples=1000, img_size=64):
    """Generate synthetic X-ray-like images for demonstration."""
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Create base image (grayscale)
        img = np.random.normal(0.5, 0.1, (img_size, img_size))
        
        # Add lung-like shapes
        center_y, center_x = img_size // 2, img_size // 2
        for dy, dx in [(-10, -12), (-10, 12)]:  # Two lung regions
            cy, cx = center_y + dy, center_x + dx
            Y, X_grid = np.ogrid[:img_size, :img_size]
            mask = ((Y - cy)**2 / 200 + (X_grid - cx)**2 / 100) < 1
            img[mask] += 0.2
        
        # Determine label and add patterns
        is_pneumonia = np.random.random() < 0.4  # 40% pneumonia cases
        
        if is_pneumonia:
            # Add "infiltrates" - cloudy patches
            n_patches = np.random.randint(2, 5)
            for _ in range(n_patches):
                py = np.random.randint(20, img_size - 20)
                px = np.random.randint(20, img_size - 20)
                patch_size = np.random.randint(5, 15)
                
                Y, X_grid = np.ogrid[:img_size, :img_size]
                patch_mask = ((Y - py)**2 + (X_grid - px)**2) < patch_size**2
                img[patch_mask] += np.random.uniform(0.1, 0.3)
            
            y.append(1)
        else:
            y.append(0)
        
        # Normalize and add noise
        img = np.clip(img + np.random.normal(0, 0.05, img.shape), 0, 1)
        X.append(img)
    
    X = np.array(X).reshape(-1, img_size, img_size, 1)
    y = np.array(y)
    
    print(f"Generated {n_samples} synthetic X-ray images")
    print(f"Normal: {sum(y==0)}, Pneumonia: {sum(y==1)}")
    
    return X, y

def build_simple_cnn(input_shape):
    """Build a simple CNN for X-ray classification."""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_transfer_learning_model(input_shape):
    """Build model using transfer learning (VGG16)."""
    if not TF_AVAILABLE:
        return None
    
    # Load pre-trained VGG16
    base_model = VGG16(weights='imagenet', include_top=False, 
                       input_shape=(input_shape[0], input_shape[1], 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20):
    """Train the model with data augmentation."""
    if model is None:
        return None
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    if model is None:
        # Dummy evaluation for when TF is not available
        y_pred = np.random.randint(0, 2, len(y_test))
        y_pred_proba = np.random.random(len(y_test))
    else:
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = np.mean(y_pred == y_test)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_results(X_test, y_test, results, history=None):
    """Plot comprehensive results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Sample images
    for i in range(4):
        ax_idx = i // 2, i % 2
        if i < 2:
            idx = np.where(y_test == 0)[0][i]
            title = 'Normal'
        else:
            idx = np.where(y_test == 1)[0][i-2]
            title = 'Pneumonia'
        
        if i == 0:
            axes[0, 0].imshow(X_test[idx].squeeze(), cmap='gray')
            axes[0, 0].set_title(f'Sample: {title}')
            axes[0, 0].axis('off')
    
    # Show more samples in a grid
    fig2, axes2 = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes2.flat):
        if i < 4:
            idx = np.where(y_test == 0)[0][i] if i < len(np.where(y_test == 0)[0]) else 0
            label = 'Normal'
        else:
            idx = np.where(y_test == 1)[0][i-4] if i-4 < len(np.where(y_test == 1)[0]) else 0
            label = 'Pneumonia'
        ax.imshow(X_test[idx].squeeze(), cmap='gray')
        ax.set_title(f'{label}\nPred: {results["probabilities"][idx]:.2f}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_xrays.png', dpi=150)
    plt.close()
    
    # 2. Training history (if available)
    if history:
        axes[0, 1].plot(history.history['accuracy'], label='Train')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Training history\nnot available', 
                        ha='center', va='center', fontsize=12)
        axes[0, 1].set_title('Model Accuracy')
    
    # 3. ROC Curve
    axes[0, 2].plot(results['fpr'], results['tpr'], 
                    label=f'AUC = {results["auc"]:.3f}', color='blue', linewidth=2)
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 2].set_title('ROC Curve')
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    import seaborn as sns
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'], ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 5. Metrics bar chart
    metrics = ['precision', 'recall', 'f1-score']
    normal_scores = [results['report']['Normal'][m] for m in metrics]
    pneumonia_scores = [results['report']['Pneumonia'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, normal_scores, width, label='Normal', color='#4ECDC4')
    axes[1, 1].bar(x + width/2, pneumonia_scores, width, label='Pneumonia', color='#FF6B6B')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_title('Classification Metrics')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Prediction distribution
    axes[1, 2].hist(results['probabilities'][y_test==0], bins=20, alpha=0.7, 
                    label='Normal', color='blue')
    axes[1, 2].hist(results['probabilities'][y_test==1], bins=20, alpha=0.7, 
                    label='Pneumonia', color='red')
    axes[1, 2].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[1, 2].set_title('Prediction Probability Distribution')
    axes[1, 2].set_xlabel('Predicted Probability')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('pneumonia_detection_results.png', dpi=150)
    plt.close()
    print("Results saved to 'pneumonia_detection_results.png'")

def main():
    print("=" * 50)
    print("Day 26: Medical Diagnosis AI - Pneumonia Detection")
    print("=" * 50)
    
    IMG_SIZE = 64
    
    # Generate data
    print("\n[1] Generating synthetic X-ray data...")
    X, y = generate_synthetic_xray_data(n_samples=2000, img_size=IMG_SIZE)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Build model
    print("\n[2] Building CNN model...")
    if TF_AVAILABLE:
        model = build_simple_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1))
        model.summary()
    else:
        print("TensorFlow not available. Using dummy predictions.")
        model = None
    
    # Train model
    print("\n[3] Training model...")
    if model:
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=15)
    else:
        history = None
    
    # Evaluate
    print("\n[4] Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  AUC: {results['auc']:.3f}")
    print(f"  Pneumonia Recall: {results['report']['Pneumonia']['recall']:.2%}")
    print(f"  Pneumonia Precision: {results['report']['Pneumonia']['precision']:.2%}")
    
    # Plot results
    print("\n[5] Generating visualizations...")
    plot_results(X_test, y_test, results, history)
    
    # Save results
    output = {
        'accuracy': float(results['accuracy']),
        'auc': float(results['auc']),
        'pneumonia_recall': float(results['report']['Pneumonia']['recall']),
        'pneumonia_precision': float(results['report']['Pneumonia']['precision']),
        'test_samples': len(X_test)
    }
    
    with open('diagnosis_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Results saved to 'diagnosis_results.json'")
    
    print("\n⚠️  DISCLAIMER: This is for educational purposes only.")
    print("    Not for actual medical diagnosis!")
    
    print("\n" + "=" * 50)
    print("Day 26 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
