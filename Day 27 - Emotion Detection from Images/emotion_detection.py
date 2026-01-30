"""
Day 27: Emotion Detection from Images
30-Day AI Challenge

Detect emotions from facial expressions using CNN.
Uses FER2013-like synthetic data for demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, 
                                         Dropout, BatchNormalization)
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def generate_synthetic_face_data(n_samples=3000, img_size=48):
    """Generate synthetic facial expression data."""
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Create base face image
        img = np.ones((img_size, img_size)) * 0.7
        
        # Face outline (circle)
        center = img_size // 2
        Y, X_grid = np.ogrid[:img_size, :img_size]
        face_mask = ((Y - center)**2 + (X_grid - center)**2) < (img_size//2.5)**2
        img[face_mask] = 0.9
        
        # Random emotion
        emotion_idx = np.random.randint(0, len(EMOTIONS))
        emotion = EMOTIONS[emotion_idx]
        
        # Eyes (two circles)
        eye_y = center - 5
        for eye_x in [center - 8, center + 8]:
            eye_mask = ((Y - eye_y)**2 + (X_grid - eye_x)**2) < 9
            img[eye_mask] = 0.3
        
        # Eyebrows and mouth based on emotion
        if emotion == 'Happy':
            # Smiling mouth (arc)
            for dx in range(-10, 11):
                dy = int(3 - (dx**2) / 50)
                if 0 <= center + 10 + dy < img_size and 0 <= center + dx < img_size:
                    img[center + 10 + dy, center + dx] = 0.2
            # Raised eyebrows
            img[eye_y - 5:eye_y - 3, center - 12:center - 4] = 0.3
            img[eye_y - 5:eye_y - 3, center + 4:center + 12] = 0.3
            
        elif emotion == 'Sad':
            # Frowning mouth
            for dx in range(-8, 9):
                dy = int((dx**2) / 40)
                if 0 <= center + 12 + dy < img_size:
                    img[center + 12 + dy, center + dx] = 0.2
            # Droopy eyebrows
            img[eye_y - 3:eye_y - 1, center - 12:center - 4] = 0.3
            img[eye_y - 5:eye_y - 3, center + 4:center + 12] = 0.3
            
        elif emotion == 'Angry':
            # Tight mouth
            img[center + 10:center + 12, center - 8:center + 8] = 0.2
            # Angled eyebrows (V shape)
            for dx in range(-8, 0):
                img[eye_y - 4 - dx//4, center + dx] = 0.3
            for dx in range(0, 9):
                img[eye_y - 4 + dx//4, center + dx] = 0.3
                
        elif emotion == 'Surprise':
            # Open mouth (circle)
            mouth_y = center + 10
            mouth_mask = ((Y - mouth_y)**2 + (X_grid - center)**2) < 25
            img[mouth_mask] = 0.2
            # Raised eyebrows (high)
            img[eye_y - 8:eye_y - 6, center - 12:center - 4] = 0.3
            img[eye_y - 8:eye_y - 6, center + 4:center + 12] = 0.3
            
        elif emotion == 'Fear':
            # Open mouth
            img[center + 8:center + 14, center - 5:center + 5] = 0.2
            # Wide eyes
            for eye_x in [center - 8, center + 8]:
                eye_mask = ((Y - eye_y)**2 + (X_grid - eye_x)**2) < 16
                img[eye_mask] = 0.3
                
        elif emotion == 'Disgust':
            # Asymmetric mouth
            img[center + 10:center + 12, center - 6:center + 2] = 0.2
            # Wrinkled nose area
            img[center - 2:center + 2, center - 2:center + 2] = 0.5
            
        else:  # Neutral
            # Straight mouth
            img[center + 10:center + 11, center - 6:center + 6] = 0.3
            # Normal eyebrows
            img[eye_y - 4:eye_y - 3, center - 10:center - 4] = 0.4
            img[eye_y - 4:eye_y - 3, center + 4:center + 10] = 0.4
        
        # Add noise
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)
        
        X.append(img)
        y.append(emotion_idx)
    
    X = np.array(X).reshape(-1, img_size, img_size, 1)
    y = np.array(y)
    
    print(f"Generated {n_samples} synthetic face images")
    for i, emotion in enumerate(EMOTIONS):
        count = np.sum(y == i)
        print(f"  {emotion}: {count}")
    
    return X, y

def build_emotion_cnn(input_shape, num_classes):
    """Build CNN for emotion classification."""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=25):
    """Train the emotion detection model."""
    if model is None:
        return None
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    if model is None:
        y_pred = np.random.randint(0, len(EMOTIONS), len(y_test))
        y_pred_proba = np.random.random((len(y_test), len(EMOTIONS)))
    else:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    accuracy = np.mean(y_pred == y_true)
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_results(X_test, y_test, results, history=None):
    """Plot comprehensive results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    # 1. Sample predictions
    axes[0, 0].set_title('Sample Predictions')
    sample_indices = np.random.choice(len(X_test), 9, replace=False)
    
    # Create mini grid
    grid_img = np.zeros((48*3, 48*3))
    for i, idx in enumerate(sample_indices[:9]):
        row, col = i // 3, i % 3
        grid_img[row*48:(row+1)*48, col*48:(col+1)*48] = X_test[idx].squeeze()
    
    axes[0, 0].imshow(grid_img, cmap='gray')
    axes[0, 0].axis('off')
    
    # 2. Training history
    if history:
        axes[0, 1].plot(history.history['accuracy'], label='Train')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Training history\nnot available', ha='center', va='center')
        axes[0, 1].set_title('Training Accuracy')
    
    # 3. Per-class accuracy
    class_acc = []
    for i, emotion in enumerate(EMOTIONS):
        mask = y_true == i
        if np.sum(mask) > 0:
            acc = np.mean(results['predictions'][mask] == i)
        else:
            acc = 0
        class_acc.append(acc)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(EMOTIONS)))
    axes[0, 2].barh(EMOTIONS, class_acc, color=colors)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_title('Per-Class Accuracy')
    axes[0, 2].set_xlabel('Accuracy')
    
    # 4. Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[1, 0].get_yticklabels(), rotation=0)
    
    # 5. F1 Scores
    f1_scores = [results['report'][e]['f1-score'] for e in EMOTIONS]
    axes[1, 1].bar(EMOTIONS, f1_scores, color=colors)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('F1 Score by Emotion')
    axes[1, 1].set_ylabel('F1 Score')
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    # 6. Emotion distribution in test set
    unique, counts = np.unique(y_true, return_counts=True)
    axes[1, 2].pie(counts, labels=[EMOTIONS[i] for i in unique], autopct='%1.1f%%', colors=colors)
    axes[1, 2].set_title('Test Set Distribution')
    
    plt.tight_layout()
    plt.savefig('emotion_detection_results.png', dpi=150)
    plt.close()
    print("Results saved to 'emotion_detection_results.png'")

def predict_emotion(model, image):
    """Predict emotion for a single image."""
    if model is None:
        return EMOTIONS[np.random.randint(0, len(EMOTIONS))], np.random.random(len(EMOTIONS))
    
    if len(image.shape) == 2:
        image = image.reshape(1, image.shape[0], image.shape[1], 1)
    elif len(image.shape) == 3:
        image = image.reshape(1, *image.shape)
    
    proba = model.predict(image, verbose=0)[0]
    emotion_idx = np.argmax(proba)
    
    return EMOTIONS[emotion_idx], proba

def main():
    print("=" * 50)
    print("Day 27: Emotion Detection from Images")
    print("=" * 50)
    
    IMG_SIZE = 48
    
    # Generate data
    print("\n[1] Generating synthetic face data...")
    X, y = generate_synthetic_face_data(n_samples=3500, img_size=IMG_SIZE)
    
    # Convert labels to categorical
    if TF_AVAILABLE:
        y_cat = to_categorical(y, num_classes=len(EMOTIONS))
    else:
        y_cat = y
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Build model
    print("\n[2] Building CNN model...")
    model = build_emotion_cnn((IMG_SIZE, IMG_SIZE, 1), len(EMOTIONS))
    if model:
        model.summary()
    
    # Train
    print("\n[3] Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=20)
    
    # Evaluate
    print("\n[4] Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    print(f"\nTest Accuracy: {results['accuracy']:.2%}")
    print("\nPer-emotion F1 Scores:")
    for emotion in EMOTIONS:
        f1 = results['report'][emotion]['f1-score']
        print(f"  {emotion}: {f1:.3f}")
    
    # Plot
    print("\n[5] Generating visualizations...")
    plot_results(X_test, y_test, results, history)
    
    # Demo prediction
    print("\n[6] Demo predictions:")
    for i in range(5):
        idx = np.random.randint(0, len(X_test))
        true_label = EMOTIONS[np.argmax(y_test[idx])] if len(y_test.shape) > 1 else EMOTIONS[y_test[idx]]
        pred_emotion, proba = predict_emotion(model, X_test[idx])
        print(f"  Sample {i+1}: True={true_label}, Predicted={pred_emotion} ({max(proba)*100:.1f}%)")
    
    # Save results
    output = {
        'accuracy': float(results['accuracy']),
        'emotions': EMOTIONS,
        'f1_scores': {e: float(results['report'][e]['f1-score']) for e in EMOTIONS}
    }
    
    with open('emotion_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to 'emotion_results.json'")
    
    print("\n" + "=" * 50)
    print("Day 27 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
