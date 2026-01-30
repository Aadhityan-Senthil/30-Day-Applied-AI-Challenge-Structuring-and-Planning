# Day 24: Time Series Forecasting for Temperature Data

## Overview
Forecast temperature using multiple time series methods and compare their performance.

## Methods Implemented
- **Moving Average**: Simple rolling mean baseline
- **Exponential Smoothing**: Weighted average with decay
- **ARIMA**: Autoregressive model (simplified)
- **LSTM**: Deep learning approach

## Requirements
```bash
pip install numpy pandas matplotlib scikit-learn
# Optional for LSTM:
pip install tensorflow
```

## Usage
```bash
python time_series_forecasting.py
```

## Data
Synthetic temperature data with:
- Yearly seasonality
- Weekly patterns
- Long-term trend
- Random noise

## Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

## Output Files
- `temperature_forecast_results.png` - Visualizations
- `forecast_results.json` - Model metrics

## Use Real Data
```python
# Download from NOAA or other sources
import pandas as pd
df = pd.read_csv('temperature_data.csv', parse_dates=['date'])
```
