# Day 20: Fake News Detection using Machine Learning

## Overview
Classify news articles as real or fake using NLP and machine learning, comparing multiple classifiers.

## Features
- Text preprocessing and cleaning
- TF-IDF vectorization with bigrams
- Multiple classifier comparison
- Feature importance analysis
- Fake news indicator extraction

## Classifiers Used
- Logistic Regression
- Passive Aggressive Classifier
- Multinomial Naive Bayes
- Random Forest

## Requirements
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage
```bash
python fake_news_detection.py
```

## Extend with Real Dataset
Download from Kaggle:
```
https://www.kaggle.com/c/fake-news/data
```

## Fake News Indicators Detected
- Sensationalist language (SHOCKING, BREAKING)
- Excessive punctuation (!!!)
- All caps words
- Conspiracy-related keywords
- Lack of sources/citations

## Output Files
- `fake_news_detection_results.png` - Model comparison and features
- `fake_news_model.pkl` - Trained model for inference

## Predict New Headlines
```python
with open('fake_news_model.pkl', 'rb') as f:
    data = pickle.load(f)
vec = data['vectorizer'].transform(["Your headline"])
prediction = data['model'].predict(vec)
```
