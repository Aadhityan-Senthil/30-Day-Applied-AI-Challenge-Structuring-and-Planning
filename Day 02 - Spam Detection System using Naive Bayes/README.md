# Day 02: Spam Detection Model with Naïve Bayes

## Overview
Build a spam email/SMS classifier using TF-IDF features and Multinomial Naive Bayes. Includes text preprocessing, cross-validation, and model persistence.

## Features
- Text preprocessing (lowercase, stemming, stopword removal)
- TF-IDF vectorization with bigrams
- Multinomial Naive Bayes classification
- 5-fold cross-validation
- Model serialization with pickle

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

## Usage
```bash
python spam_detection.py
```

## Dataset
Download from Kaggle or use built-in sample data:
```
https://www.kaggle.com/uciml/sms-spam-collection-dataset
```

Place `spam.csv` in the same directory, or the script will use sample data.

## Model Pipeline
1. Text preprocessing (clean, stem, remove stopwords)
2. TF-IDF vectorization (max 5000 features, bigrams)
3. Multinomial Naive Bayes (alpha=0.1)

## Output Files
- `spam_detection_results.png` - Confusion matrix and metrics
- `spam_classifier.pkl` - Trained model
- `model_results.json` - Performance metrics

## Key Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | True positives / Predicted positives |
| Recall | True positives / Actual positives |
| F1-Score | Harmonic mean of precision and recall |

## Use Saved Model
```python
import pickle
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
prediction = model.predict(["Your message here"])
```

## Key Learnings
- TF-IDF captures word importance across documents
- Naive Bayes works well for text classification
- Preprocessing significantly impacts performance
- Class imbalance affects precision/recall tradeoff
