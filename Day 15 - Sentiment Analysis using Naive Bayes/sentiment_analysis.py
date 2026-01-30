"""
Day 15: Sentiment Analysis using Naive Bayes
30-Day AI Challenge

Build a sentiment classifier to analyze text as positive or negative
using the Naive Bayes algorithm with IMDB dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re
import pickle

# Sample dataset (expandable - you can download IMDB dataset for more data)
SAMPLE_DATA = [
    # Positive reviews
    ("This movie is absolutely wonderful and amazing!", 1),
    ("I love this product, it works perfectly!", 1),
    ("Best purchase I've ever made, highly recommend!", 1),
    ("Fantastic experience, will definitely come back!", 1),
    ("The food was delicious and service was excellent!", 1),
    ("Great quality and fast shipping, very happy!", 1),
    ("This book changed my life, must read!", 1),
    ("Amazing customer support, they helped me right away!", 1),
    ("Perfect fit and great style, love it!", 1),
    ("Exceeded my expectations in every way!", 1),
    ("Such a beautiful place, highly recommend visiting!", 1),
    ("The app is user-friendly and works great!", 1),
    ("Incredible value for money, very satisfied!", 1),
    ("The team did an outstanding job!", 1),
    ("I'm so happy with this purchase!", 1),
    ("Wonderful experience from start to finish!", 1),
    ("The quality is superb, worth every penny!", 1),
    ("Absolutely loved every minute of it!", 1),
    ("Best decision I ever made!", 1),
    ("Everything was perfect, no complaints!", 1),
    ("Really enjoyed this film, great acting!", 1),
    ("Five stars, couldn't be happier!", 1),
    ("Brilliant work, highly impressive!", 1),
    ("Outstanding performance by everyone!", 1),
    ("This made my day so much better!", 1),
    
    # Negative reviews
    ("This is the worst product ever!", 0),
    ("Terrible experience, never going back!", 0),
    ("Complete waste of money, do not buy!", 0),
    ("Very disappointed with the quality!", 0),
    ("Poor customer service, very rude staff!", 0),
    ("The food was cold and tasteless!", 0),
    ("Broke after one week, cheaply made!", 0),
    ("Slow delivery and damaged package!", 0),
    ("Not as described, total scam!", 0),
    ("Horrible experience, asked for refund!", 0),
    ("The app crashes constantly, useless!", 0),
    ("Overpriced and underwhelming!", 0),
    ("Would give zero stars if I could!", 0),
    ("Stay away from this place!", 0),
    ("Regret buying this, waste of time!", 0),
    ("The worst movie I've ever seen!", 0),
    ("Extremely disappointed, not recommended!", 0),
    ("Poor quality and bad customer support!", 0),
    ("This product is a complete disaster!", 0),
    ("Never buying from them again!", 0),
    ("Awful experience, totally frustrated!", 0),
    ("Don't waste your money on this!", 0),
    ("Terrible quality, fell apart immediately!", 0),
    ("Worst purchase of my life!", 0),
    ("Absolutely horrible, avoid at all costs!", 0),
]

def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def create_dataset():
    """Create dataset from sample data."""
    texts = [preprocess_text(text) for text, _ in SAMPLE_DATA]
    labels = [label for _, label in SAMPLE_DATA]
    
    print(f"Dataset size: {len(texts)} samples")
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    return texts, labels

def train_model(X_train, y_train):
    """Train Naive Bayes classifier with TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train_vec, y_train)
    
    return classifier, vectorizer

def evaluate_model(classifier, vectorizer, X_test, y_test):
    """Evaluate model performance."""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm

def get_top_features(classifier, vectorizer, n=10):
    """Get most indicative words for each sentiment."""
    feature_names = vectorizer.get_feature_names_out()
    
    neg_idx = np.argsort(classifier.feature_log_prob_[0])[-n:][::-1]
    pos_idx = np.argsort(classifier.feature_log_prob_[1])[-n:][::-1]
    
    neg_words = [feature_names[i] for i in neg_idx]
    pos_words = [feature_names[i] for i in pos_idx]
    
    return neg_words, pos_words

def predict_sentiment(classifier, vectorizer, text):
    """Predict sentiment for new text."""
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    prediction = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]
    
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = max(proba) * 100
    
    return sentiment, confidence

def plot_results(cm, neg_words, pos_words, accuracy):
    """Visualize results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Negative words
    axes[1].barh(range(len(neg_words)), range(len(neg_words), 0, -1), color='red', alpha=0.7)
    axes[1].set_yticks(range(len(neg_words)))
    axes[1].set_yticklabels(neg_words)
    axes[1].set_title('Top Negative Indicator Words')
    axes[1].set_xlabel('Importance')
    
    # Positive words
    axes[2].barh(range(len(pos_words)), range(len(pos_words), 0, -1), color='green', alpha=0.7)
    axes[2].set_yticks(range(len(pos_words)))
    axes[2].set_yticklabels(pos_words)
    axes[2].set_title('Top Positive Indicator Words')
    axes[2].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('sentiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Results saved to 'sentiment_results.png'")

def main():
    print("=" * 50)
    print("Day 15: Sentiment Analysis using Naive Bayes")
    print("=" * 50)
    
    # Create dataset
    print("\n[1] Creating dataset...")
    texts, labels = create_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    print("\n[2] Training Naive Bayes classifier...")
    classifier, vectorizer = train_model(X_train, y_train)
    
    # Evaluate
    print("\n[3] Evaluating model...")
    accuracy, report, cm = evaluate_model(classifier, vectorizer, X_test, y_test)
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"\nClassification Report:\n{report}")
    
    # Get top features
    neg_words, pos_words = get_top_features(classifier, vectorizer)
    print(f"\nTop Negative Words: {', '.join(neg_words)}")
    print(f"Top Positive Words: {', '.join(pos_words)}")
    
    # Plot results
    print("\n[4] Generating visualizations...")
    plot_results(cm, neg_words, pos_words, accuracy)
    
    # Demo predictions
    print("\n[5] Demo Predictions:")
    test_sentences = [
        "I absolutely love this product!",
        "This is terrible, complete waste of money.",
        "Pretty good overall, satisfied with purchase.",
        "Worst experience ever, very disappointed."
    ]
    
    for sentence in test_sentences:
        sentiment, confidence = predict_sentiment(classifier, vectorizer, sentence)
        print(f"  '{sentence[:40]}...' → {sentiment} ({confidence:.1f}%)")
    
    # Save model
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump({'classifier': classifier, 'vectorizer': vectorizer}, f)
    print("\nModel saved to 'sentiment_model.pkl'")
    
    print("\n" + "=" * 50)
    print("Day 15 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
