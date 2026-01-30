# Day 04: Loan Approval Predictor using Random Forest

## Overview
Build a loan approval prediction model using Random Forest classifier with preprocessing pipelines, hyperparameter tuning, and comprehensive evaluation metrics.

## Features
- ColumnTransformer for mixed data types
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- GridSearchCV for hyperparameter tuning
- ROC-AUC analysis

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
```bash
python loan_approval_predictor.py
```

## Dataset
Place `loan_data.csv` in the same directory, or sample data will be generated.

## Features Used
| Feature | Type | Description |
|---------|------|-------------|
| person_age | Numeric | Applicant's age |
| person_income | Numeric | Annual income |
| loan_amnt | Numeric | Loan amount requested |
| loan_int_rate | Numeric | Interest rate |
| person_home_ownership | Categorical | RENT/OWN/MORTGAGE |
| loan_intent | Categorical | Purpose of loan |

## Model Pipeline
1. ColumnTransformer for preprocessing
2. RandomForestClassifier with GridSearchCV
3. Stratified train-test split

## Output Files
- `loan_approval_results.png` - Confusion matrix, ROC curve, feature importance
- `model_results.json` - Performance metrics

## Key Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Approved loans that should be approved
- **Recall**: Catching all good loan candidates
- **ROC-AUC**: Model discrimination ability

## Key Learnings
- Pipelines ensure consistent preprocessing
- Random Forest handles mixed feature types well
- Feature importance reveals lending criteria
- ROC-AUC better than accuracy for imbalanced data
