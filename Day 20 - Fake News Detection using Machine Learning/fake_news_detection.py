"""
Day 20: Fake News Detection using Machine Learning
30-Day AI Challenge

Build a classifier to detect fake news articles using NLP and ML.
Uses TF-IDF features with various classifiers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import pickle
import json

# Sample dataset (Real news = 1, Fake news = 0)
SAMPLE_DATA = [
    # Real news examples
    ("Scientists at NASA have confirmed the discovery of water ice on Mars surface using data from the Mars Reconnaissance Orbiter.", 1),
    ("The Federal Reserve announced a 0.25 percentage point interest rate increase, citing ongoing inflation concerns.", 1),
    ("Apple Inc reported quarterly revenue of $89.5 billion, slightly below analyst expectations.", 1),
    ("The World Health Organization declared the end of the global health emergency for COVID-19.", 1),
    ("Researchers published a peer-reviewed study in Nature showing promising results for a new cancer treatment.", 1),
    ("The United Nations Security Council held an emergency session to discuss the ongoing humanitarian crisis.", 1),
    ("Tesla delivered 435,000 vehicles in Q3, according to the company's official quarterly report.", 1),
    ("The European Central Bank maintained interest rates at current levels following their monthly meeting.", 1),
    ("A new species of deep-sea fish was discovered by marine biologists during an expedition.", 1),
    ("The unemployment rate fell to 3.7% according to the Bureau of Labor Statistics monthly report.", 1),
    ("Microsoft announced a partnership with OpenAI to develop new artificial intelligence technologies.", 1),
    ("The Supreme Court issued a ruling on the landmark case after months of deliberation.", 1),
    ("Scientists successfully tested a new vaccine candidate in phase 3 clinical trials.", 1),
    ("Amazon reported strong growth in its cloud computing division AWS during earnings call.", 1),
    ("The International Monetary Fund released its updated global economic outlook report.", 1),
    
    # Fake news examples
    ("SHOCKING: Celebrity secretly controls world government from underground bunker!", 0),
    ("Scientists EXPOSED: Climate change is a hoax created by the liberal elite!", 0),
    ("BREAKING: Aliens have been living among us for decades, government insider reveals!", 0),
    ("You won't BELIEVE what this politician said about mind control vaccines!", 0),
    ("URGENT: Share this before they delete it - the truth about chemtrails!", 0),
    ("Secret cure for cancer has been hidden by big pharma for 50 years!", 0),
    ("EXPOSED: The moon landing was filmed in a Hollywood studio, new evidence proves!", 0),
    ("Miracle weight loss pill lets you lose 30 pounds in ONE WEEK with no exercise!", 0),
    ("SHOCKING revelation: 5G towers are actually mind control devices!", 0),
    ("Government planning to implant microchips in all citizens by next year!", 0),
    ("Famous billionaire admits the Earth is actually FLAT in leaked interview!", 0),
    ("BREAKING: Time travel has been achieved but kept secret from the public!", 0),
    ("Scientists discover that drinking bleach cures all diseases - doctors hate this!", 0),
    ("SECRET document reveals reptilian aliens control major world banks!", 0),
    ("URGENT: Forward this message to 10 people or bad luck will follow!", 0),
]

def preprocess_text(text):
    """Clean and preprocess text for analysis."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_features(texts):
    """Extract additional features from text."""
    features = []
    
    for text in texts:
        feature_dict = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'has_shocking': 1 if 'shocking' in text.lower() else 0,
            'has_breaking': 1 if 'breaking' in text.lower() else 0,
            'has_urgent': 1 if 'urgent' in text.lower() else 0,
            'has_exposed': 1 if 'exposed' in text.lower() else 0,
        }
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def create_dataset():
    """Create and prepare the dataset."""
    texts = [text for text, _ in SAMPLE_DATA]
    labels = [label for _, label in SAMPLE_DATA]
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    df = pd.DataFrame({
        'original_text': texts,
        'processed_text': processed_texts,
        'label': labels,
        'label_name': ['Real' if l == 1 else 'Fake' for l in labels]
    })
    
    print(f"Dataset size: {len(df)}")
    print(f"Real news: {sum(labels)}")
    print(f"Fake news: {len(labels) - sum(labels)}")
    
    return df

def train_classifiers(X_train, y_train):
    """Train multiple classifiers and return them."""
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Passive Aggressive': PassiveAggressiveClassifier(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        print(f"  Trained: {name}")
    
    return trained_models

def evaluate_classifiers(models, X_test, y_test, vectorizer):
    """Evaluate all trained classifiers."""
    results = {}
    
    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  F1 (Fake): {report['Fake']['f1-score']:.2f}")
        print(f"  F1 (Real): {report['Real']['f1-score']:.2f}")
    
    return results

def get_important_features(vectorizer, model, n=15):
    """Get most important features for fake vs real classification."""
    if not hasattr(model, 'coef_'):
        return None, None
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Top fake indicators (negative coefficients)
    fake_idx = np.argsort(coefficients)[:n]
    fake_features = [(feature_names[i], coefficients[i]) for i in fake_idx]
    
    # Top real indicators (positive coefficients)
    real_idx = np.argsort(coefficients)[-n:][::-1]
    real_features = [(feature_names[i], coefficients[i]) for i in real_idx]
    
    return fake_features, real_features

def plot_results(results, fake_features, real_features):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Accuracy comparison
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    
    axes[0, 0].barh(names, accuracies, color=colors)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    for i, v in enumerate(accuracies):
        axes[0, 0].text(v + 0.01, i, f'{v:.2%}', va='center')
    
    # 2. Confusion matrix (best model)
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    cm = results[best_model]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    axes[0, 1].set_title(f'Confusion Matrix ({best_model})')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Top fake indicators
    if fake_features:
        words, scores = zip(*fake_features[:10])
        axes[1, 0].barh(words, np.abs(scores), color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top Fake News Indicators')
        axes[1, 0].invert_yaxis()
    
    # 4. Top real indicators
    if real_features:
        words, scores = zip(*real_features[:10])
        axes[1, 1].barh(words, scores, color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top Real News Indicators')
        axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('fake_news_detection_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nResults saved to 'fake_news_detection_results.png'")

def predict_news(text, vectorizer, model):
    """Predict if a news article is fake or real."""
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)
    else:
        confidence = None
    
    result = "REAL ✓" if prediction == 1 else "FAKE ✗"
    return result, confidence

def main():
    print("=" * 50)
    print("Day 20: Fake News Detection")
    print("=" * 50)
    
    # Create dataset
    print("\n[1] Creating dataset...")
    df = create_dataset()
    
    # Split data
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize text
    print("\n[2] Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature matrix shape: {X_train_vec.shape}")
    
    # Train classifiers
    print("\n[3] Training classifiers...")
    models = train_classifiers(X_train_vec, y_train)
    
    # Evaluate
    print("\n[4] Evaluating models...")
    results = evaluate_classifiers(models, X_test_vec, y_test, vectorizer)
    
    # Get feature importance
    print("\n[5] Extracting important features...")
    fake_features, real_features = get_important_features(
        vectorizer, models['Logistic Regression']
    )
    
    if fake_features:
        print("\nTop Fake News Indicators:")
        for word, score in fake_features[:5]:
            print(f"  {word}: {score:.3f}")
        
        print("\nTop Real News Indicators:")
        for word, score in real_features[:5]:
            print(f"  {word}: {score:.3f}")
    
    # Plot results
    print("\n[6] Generating visualizations...")
    plot_results(results, fake_features, real_features)
    
    # Demo predictions
    print("\n[7] Demo Predictions:")
    test_headlines = [
        "Scientists discover new treatment for Alzheimer's disease in clinical trials",
        "SHOCKING: Secret government program controls weather with chemtrails!",
        "Apple announces new iPhone with improved camera features",
        "You WON'T BELIEVE what this celebrity said about vaccines!!!"
    ]
    
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_name]
    
    for headline in test_headlines:
        result, conf = predict_news(headline, vectorizer, best_model)
        conf_str = f" ({conf:.1%} confidence)" if conf else ""
        print(f"\n  '{headline[:50]}...'")
        print(f"  → {result}{conf_str}")
    
    # Save model
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'model': best_model}, f)
    print(f"\nBest model ({best_model_name}) saved to 'fake_news_model.pkl'")
    
    print("\n" + "=" * 50)
    print("Day 20 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
