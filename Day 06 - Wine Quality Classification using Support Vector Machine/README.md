# Day 06: Wine Quality Classification (Support Vector Machine)

## Overview
Classify wine quality using Support Vector Machine (SVM) with RBF kernel, hyperparameter tuning, and SHAP explainability analysis.

## Features
- SVM with RBF kernel
- GridSearchCV for C and gamma tuning
- Feature scaling with StandardScaler
- SHAP values for model interpretation
- Binary classification (Good vs Bad wine)

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap
```

## Usage
```bash
python wine_quality_classification.py
```

## Dataset
Place `WineQT.csv` in the same directory.
Download from: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

## Features Used
| Feature | Description |
|---------|-------------|
| fixed acidity | Tartaric acid content |
| volatile acidity | Acetic acid content |
| citric acid | Citric acid content |
| residual sugar | Sugar after fermentation |
| chlorides | Salt content |
| free sulfur dioxide | Free SO2 |
| total sulfur dioxide | Total SO2 |
| density | Wine density |
| pH | Acidity level |
| sulphates | Sulfate content |
| alcohol | Alcohol percentage |

## Classification
- **Good Wine**: Quality score ≥ 7
- **Bad Wine**: Quality score < 7

## Model Pipeline
1. Convert quality to binary labels
2. StandardScaler normalization
3. SVM with RBF kernel
4. GridSearchCV (C, gamma)
5. SHAP analysis

## Output Files
- Confusion matrix visualization
- SHAP summary plot
- Classification metrics

## Key Metrics
- Accuracy
- Precision / Recall
- F1-Score
- Support per class

## Key Learnings
- SVM works well with scaled features
- RBF kernel captures non-linear relationships
- C controls regularization strength
- Gamma controls RBF influence radius
- SHAP explains individual predictions
