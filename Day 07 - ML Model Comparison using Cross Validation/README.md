# Day 07: Comparing ML Models using Cross-Validation

## Overview
Compare multiple machine learning classification algorithms using cross-validation, ROC curves, and comprehensive metrics to find the best model for wine quality prediction.

## Features
- 7 different classifiers compared
- 5-fold cross-validation
- ROC curve analysis
- Hyperparameter tuning for best model
- Visual model comparison

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage
```bash
python model_comparison.py
```

## Dataset
Uses `WineQT.csv` from Day 06.

## Models Compared
| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential tree boosting |
| SVM | Support Vector Machine |
| K-Nearest Neighbors | Distance-based classifier |
| Decision Tree | Single tree classifier |
| XGBoost | Extreme Gradient Boosting |

## Evaluation Process
1. Load and preprocess data
2. Feature scaling with StandardScaler
3. 5-fold cross-validation for each model
4. Select best model based on CV accuracy
5. Hyperparameter tuning for winner
6. Final evaluation on test set

## Output Files
- `model_comparison.png` - Bar chart of CV accuracies
- ROC curve for best model
- Confusion matrix
- Classification report

## Key Metrics
- Cross-validation accuracy (mean ± std)
- Test accuracy
- ROC-AUC score
- Precision, Recall, F1

## Typical Results
Models are ranked by CV accuracy. XGBoost and Random Forest often perform best on tabular data.

## Key Learnings
- Cross-validation provides reliable estimates
- Ensemble methods often outperform single models
- Model selection should consider multiple metrics
- Hyperparameter tuning improves performance
- No single model is best for all problems
