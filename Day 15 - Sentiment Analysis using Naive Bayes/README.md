# Day 15: Sentiment Analysis using Naive Bayes

## Overview
Text sentiment classifier that determines if text is positive or negative using Multinomial Naive Bayes with TF-IDF features.

## Features
- Text preprocessing (lowercase, remove punctuation)
- TF-IDF vectorization with bigrams
- Multinomial Naive Bayes classification
- Feature importance visualization
- Model persistence with pickle

## Requirements
```bash
pip install numpy scikit-learn matplotlib seaborn
```

## Usage
```bash
python sentiment_analysis.py
```

## Extend with Real Dataset
Download IMDB dataset:
```python
from sklearn.datasets import fetch_20newsgroups
# Or use: https://ai.stanford.edu/~amaas/data/sentiment/
```

## Output Files
- `sentiment_results.png` - Confusion matrix and feature visualization
- `sentiment_model.pkl` - Trained model for inference

## How It Works
1. **Preprocessing**: Clean text, remove noise
2. **Vectorization**: Convert text to TF-IDF features
3. **Training**: Naive Bayes learns word probabilities
4. **Prediction**: Calculate posterior probability for each class
