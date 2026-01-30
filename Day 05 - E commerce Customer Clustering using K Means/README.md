# Day 05: Energy Efficiency Prediction (Multi-Output Regression)

## Overview
Predict building heating and cooling loads simultaneously using multi-output regression with Gradient Boosting and MLP Neural Networks.

## Features
- Multi-output regression for two targets
- Gradient Boosting Regressor
- MLP Neural Network comparison
- Feature importance analysis
- Error distribution visualization

## Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

## Usage
```bash
python energy_efficiency_prediction.py
```

## Dataset
Automatically downloaded from UCI ML Repository:
- **Source**: Energy Efficiency Dataset
- **Samples**: 768 buildings
- **Features**: 8 building characteristics
- **Targets**: Heating Load, Cooling Load

## Features Used
| Feature | Description |
|---------|-------------|
| Relative Compactness | Shape compactness |
| Surface Area | Total surface area |
| Wall Area | Wall surface area |
| Roof Area | Roof surface area |
| Overall Height | Building height |
| Orientation | Building orientation |
| Glazing Area | Window area |
| Glazing Area Distribution | Window distribution |

## Models Compared
1. **Gradient Boosting**: Tree-based ensemble
2. **MLP Regressor**: Neural network

## Output Files
- Visualizations showing feature importance
- Error distribution plots
- Model comparison metrics

## Key Metrics
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Square Error |
| R² | Coefficient of Determination |

## Key Learnings
- Multi-output regression handles correlated targets
- Gradient Boosting often outperforms neural networks on tabular data
- Feature importance reveals key building characteristics
- Error analysis helps understand model weaknesses
