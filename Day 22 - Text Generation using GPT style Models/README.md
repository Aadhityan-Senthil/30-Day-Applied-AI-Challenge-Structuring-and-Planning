# Day 22: Stock Price Prediction using LSTM

## Overview
Predict stock prices using LSTM neural networks with real data from Yahoo Finance.

## Features
- Real stock data download (yfinance)
- LSTM with dropout regularization
- Future price forecasting
- Performance visualization
- Synthetic data fallback

## Requirements
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
```

## Usage
```bash
python stock_prediction.py
```

## Configuration
Modify these variables in `main()`:
- `TICKER`: Stock symbol (default: "AAPL")
- `LOOKBACK`: Days of history for prediction (default: 60)
- `EPOCHS`: Training epochs (default: 50)

## Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

## Output Files
- `stock_prediction_results.png` - Visualizations
- `prediction_results.json` - Predictions data

## ⚠️ Disclaimer
This is for **educational purposes only**. Stock market prediction is inherently uncertain. Do NOT use for actual trading decisions.
