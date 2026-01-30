"""
Day 22: Stock Price Prediction using LSTM
30-Day AI Challenge

Predict stock prices using Long Short-Term Memory (LSTM) neural networks.
Downloads real stock data from Yahoo Finance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# Try to import tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def download_stock_data(ticker="AAPL", period="2y"):
    """Download stock data from Yahoo Finance."""
    if not YF_AVAILABLE:
        print("yfinance not installed. Generating synthetic data...")
        return generate_synthetic_stock_data()
    
    print(f"Downloading {ticker} data...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError("Empty data")
        print(f"Downloaded {len(df)} days of data")
        return df
    except:
        print(f"Could not download {ticker}. Using synthetic data...")
        return generate_synthetic_stock_data()

def generate_synthetic_stock_data(days=500, start_price=150):
    """Generate synthetic stock data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='D')
    
    trend = np.linspace(0, 50, days)
    noise = np.cumsum(np.random.randn(days) * 2)
    seasonality = 10 * np.sin(np.linspace(0, 8*np.pi, days))
    close_prices = start_price + trend + noise + seasonality
    close_prices = np.maximum(close_prices, 10)
    
    df = pd.DataFrame({
        'Open': close_prices * (1 + np.random.randn(days) * 0.01),
        'High': close_prices * (1 + np.abs(np.random.randn(days) * 0.02)),
        'Low': close_prices * (1 - np.abs(np.random.randn(days) * 0.02)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    print(f"Generated {len(df)} days of synthetic data")
    return df

def prepare_data(df, feature_col='Close', lookback=60):
    """Prepare data for LSTM model."""
    data = df[feature_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def split_data(X, y, train_ratio=0.8):
    """Split data into training and testing sets."""
    split_idx = int(len(X) * train_ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

def build_lstm_model(lookback, units=50):
    """Build LSTM model for price prediction."""
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """Train the LSTM model."""
    if model is None:
        return None
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, callbacks=[early_stop], verbose=1)
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model and return metrics."""
    if model is None:
        predictions = np.roll(y_test, 1)
        predictions[0] = y_test[0]
    else:
        predictions = model.predict(X_test, verbose=0).flatten()
    
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
    
    return predictions_inv, y_test_inv, {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def predict_future(model, last_sequence, scaler, days=30):
    """Predict future prices."""
    if model is None:
        last_val = scaler.inverse_transform(last_sequence[-1].reshape(-1, 1))[0, 0]
        return [last_val * (1 + np.random.randn() * 0.01) for _ in range(days)]
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)[0, 0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def plot_results(df, y_test_inv, predictions_inv, future_predictions, metrics):
    """Plot all results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(df.index, df['Close'], label='Historical', color='blue')
    axes[0, 0].set_title('Stock Price History')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(y_test_inv, label='Actual', color='blue')
    axes[0, 1].plot(predictions_inv, label='Predicted', color='red', alpha=0.7)
    axes[0, 1].set_title(f'Predictions vs Actual (RMSE: ${metrics["RMSE"]:.2f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    last_prices = list(df['Close'].values[-30:])
    axes[1, 0].plot(range(len(last_prices)), last_prices, label='Recent', color='blue')
    axes[1, 0].plot(range(len(last_prices)-1, len(last_prices) + len(future_predictions)),
                    [last_prices[-1]] + list(future_predictions),
                    label='Forecast', color='green', linestyle='--')
    axes[1, 0].set_title('30-Day Forecast')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = axes[1, 1].bar(list(metrics.keys()), list(metrics.values()), color=colors)
    axes[1, 1].set_title('Performance Metrics')
    
    plt.tight_layout()
    plt.savefig('stock_prediction_results.png', dpi=150)
    plt.close()
    print("Results saved to 'stock_prediction_results.png'")

def main():
    print("=" * 50)
    print("Day 22: Stock Price Prediction using LSTM")
    print("=" * 50)
    
    TICKER, LOOKBACK, EPOCHS = "AAPL", 60, 50
    
    print("\n[1] Downloading stock data...")
    df = download_stock_data(TICKER, period="2y")
    
    print("\n[2] Preparing data...")
    X, y, scaler = prepare_data(df, 'Close', LOOKBACK)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    print("\n[3] Building LSTM model...")
    model = build_lstm_model(LOOKBACK) if TF_AVAILABLE else None
    
    print("\n[4] Training model...")
    if model:
        train_model(model, X_train, y_train, epochs=EPOCHS)
    
    print("\n[5] Evaluating...")
    predictions_inv, y_test_inv, metrics = evaluate_model(model, X_test, y_test, scaler)
    print(f"RMSE: ${metrics['RMSE']:.2f}, MAE: ${metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
    
    print("\n[6] Forecasting 30 days...")
    future = predict_future(model, X[-1].flatten(), scaler, 30)
    print(f"Current: ${df['Close'].iloc[-1]:.2f} → Predicted: ${future[-1]:.2f}")
    
    print("\n[7] Generating plots...")
    plot_results(df, y_test_inv, predictions_inv, future, metrics)
    
    with open('prediction_results.json', 'w') as f:
        json.dump({'ticker': TICKER, 'metrics': {k: float(v) for k, v in metrics.items()},
                   'forecast': [float(p) for p in future]}, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Day 22 Complete!")
    print("⚠️  DISCLAIMER: For educational purposes only!")

if __name__ == "__main__":
    main()
