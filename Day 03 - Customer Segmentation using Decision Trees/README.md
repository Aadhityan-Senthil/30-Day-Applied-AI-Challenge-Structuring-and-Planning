# Day 03: Customer Segmentation using Decision Tree

## Overview
Segment retail customers using K-Means clustering, then build a Decision Tree classifier to predict customer segments based on demographics and spending behavior.

## Features
- K-Means clustering with elbow method
- Decision Tree with hyperparameter tuning
- Feature scaling with StandardScaler
- Tree visualization
- Segment profile analysis

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
```bash
python customer_segmentation.py
```

## Dataset
Download Mall Customers dataset from Kaggle:
```
https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial
```

Place `Mall_Customers.csv` in the same directory.

## Features Used
- **Age**: Customer age
- **Annual Income**: Income in thousands ($)
- **Spending Score**: 1-100 score based on spending behavior

## Model Pipeline
1. Load and preprocess customer data
2. Apply K-Means clustering (k=4)
3. Scale features with StandardScaler
4. Train Decision Tree with GridSearchCV
5. Visualize segments and tree

## Output Files
- `elbow_curve.png` - Optimal cluster selection
- `customer_segmentation_results.png` - Segments and metrics
- `decision_tree.png` - Tree visualization
- `model_results.json` - Parameters and metrics

## Customer Segments
| Segment | Profile |
|---------|---------|
| 0 | High income, high spending |
| 1 | Low income, low spending |
| 2 | High income, low spending |
| 3 | Low income, high spending |

## Key Learnings
- Clustering creates labels for classification
- Decision Trees are interpretable models
- Feature importance reveals key drivers
- Hyperparameter tuning prevents overfitting
