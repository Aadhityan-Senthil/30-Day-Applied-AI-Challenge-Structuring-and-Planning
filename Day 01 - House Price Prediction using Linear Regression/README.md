# Day 01: Predicting House Prices Using Linear Regression

## Overview
Build a house price prediction model using Linear Regression with polynomial feature expansion, cross-validation, and hyperparameter tuning on the California Housing dataset.

## Features
- Exploratory Data Analysis with correlation heatmap
- Feature scaling with StandardScaler
- Polynomial feature expansion (degree 2)
- Cross-validation (5-fold)
- Hyperparameter tuning with GridSearchCV
- Comprehensive visualization

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
```bash
python house_price_prediction.py
```

## Dataset
Uses scikit-learn's built-in California Housing dataset:
- **Samples**: 20,640 houses
- **Features**: 8 (MedInc, HouseAge, AveRooms, etc.)
- **Target**: Median house value (in $100k)

## Model Pipeline
1. Load California Housing dataset
2. Correlation analysis for feature selection
3. StandardScaler for normalization
4. PolynomialFeatures (degree=2) for non-linear relationships
5. LinearRegression with GridSearchCV

## Output Files
- `correlation_matrix.png` - Feature correlation heatmap
- `house_price_results.png` - Model performance visualizations
- `model_results.json` - Metrics and parameters

## Key Metrics
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error |
| R² | Coefficient of Determination |

## Key Learnings
- Feature selection improves model interpretability
- Polynomial features capture non-linear relationships
- Cross-validation prevents overfitting
- Feature scaling is essential for regression
