# Day 25: Credit Card Fraud Detection using Anomaly Detection

## Overview
Detect fraudulent credit card transactions using various anomaly detection and machine learning techniques.

## Methods Implemented
- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor (LOF)**: Density-based detection
- **One-Class SVM**: Support vector anomaly detection
- **Random Forest**: Supervised classification (baseline)

## Requirements
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage
```bash
python fraud_detection.py
```

## Features Used
- Transaction amount
- Hour of day
- Daily transaction frequency
- Distance from home location
- Derived features (log amount, night flag)

## Real Dataset
Download from Kaggle:
```
https://www.kaggle.com/mlg-ulb/creditcardfraud
```

## Output Files
- `fraud_detection_results.png` - Visualizations
- `fraud_detection_results.json` - Model metrics

## Key Metrics
- **Precision**: How many flagged transactions are actually fraud
- **Recall**: How many frauds were caught
- **F1 Score**: Balance of precision and recall
