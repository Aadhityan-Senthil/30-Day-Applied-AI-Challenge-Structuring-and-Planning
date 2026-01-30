"""
Day 24: Time Series Forecasting for Temperature Data
30-Day AI Challenge

Forecast temperature using various time series methods including
ARIMA, Prophet, and LSTM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

def generate_temperature_data(days=730):
    """Generate synthetic temperature data with seasonality."""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Components
    trend = np.linspace(0, 2, days)  # Slight warming trend
    yearly = 15 * np.sin(2 * np.pi * np.arange(days) / 365)  # Yearly seasonality
    weekly = 2 * np.sin(2 * np.pi * np.arange(days) / 7)  # Weekly pattern
    noise = np.random.normal(0, 3, days)  # Random noise
    
    # Base temperature around 15°C
    temperature = 15 + trend + yearly + weekly + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature
    })
    df.set_index('date', inplace=True)
    
    return df

def create_sequences(data, lookback=30):
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

class MovingAverageModel:
    """Simple Moving Average baseline."""
    
    def __init__(self, window=7):
        self.window = window
        
    def fit(self, data):
        self.data = data
        
    def predict(self, steps=30):
        predictions = []
        history = list(self.data[-self.window:])
        
        for _ in range(steps):
            pred = np.mean(history)
            predictions.append(pred)
            history.pop(0)
            history.append(pred)
        
        return np.array(predictions)

class ExponentialSmoothing:
    """Exponential smoothing model."""
    
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        
    def fit(self, data):
        self.data = data
        self.last_value = data[-1]
        
    def predict(self, steps=30):
        predictions = []
        current = self.last_value
        
        for _ in range(steps):
            predictions.append(current)
            # Decay towards mean
            current = self.alpha * current + (1 - self.alpha) * np.mean(self.data)
        
        return np.array(predictions)

class SimpleARIMA:
    """Simplified ARIMA-like model using autoregression."""
    
    def __init__(self, order=5):
        self.order = order
        self.coefficients = None
        
    def fit(self, data):
        """Fit AR coefficients using least squares."""
        X = []
        y = []
        
        for i in range(self.order, len(data)):
            X.append(data[i-self.order:i])
            y.append(data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Solve using least squares
        self.coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        self.history = list(data[-self.order:])
        
    def predict(self, steps=30):
        predictions = []
        history = self.history.copy()
        
        for _ in range(steps):
            pred = np.dot(history, self.coefficients)
            predictions.append(pred)
            history.pop(0)
            history.append(pred)
        
        return np.array(predictions)

def build_lstm_forecaster(data, lookback=30, epochs=50):
    """Build and train LSTM for forecasting."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except ImportError:
        return None, None
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    X, y = create_sequences(scaled, lookback)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    return model, scaler

def evaluate_forecast(actual, predicted):
    """Calculate forecast metrics."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def plot_forecast_results(df, forecasts, test_data, dates):
    """Plot forecasting results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Historical data
    axes[0, 0].plot(df.index[-365:], df['temperature'].values[-365:], color='blue')
    axes[0, 0].set_title('Temperature History (Last Year)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Forecast comparison
    axes[0, 1].plot(dates, test_data, label='Actual', color='blue', linewidth=2)
    for name, pred in forecasts.items():
        axes[0, 1].plot(dates, pred, label=name, alpha=0.7)
    axes[0, 1].set_title('Forecast Comparison')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error comparison
    metrics = {}
    for name, pred in forecasts.items():
        metrics[name] = evaluate_forecast(test_data, pred)
    
    model_names = list(metrics.keys())
    rmse_values = [metrics[m]['RMSE'] for m in model_names]
    mae_values = [metrics[m]['MAE'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, rmse_values, width, label='RMSE', color='#FF6B6B')
    axes[1, 0].bar(x + width/2, mae_values, width, label='MAE', color='#4ECDC4')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].set_title('Error Metrics Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Seasonal decomposition (simplified)
    temp = df['temperature'].values
    trend = pd.Series(temp).rolling(30).mean().values
    seasonal = temp - trend
    
    axes[1, 1].plot(df.index[-180:], seasonal[-180:], color='purple', alpha=0.7)
    axes[1, 1].set_title('Seasonal Component (Last 6 Months)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Deviation (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_forecast_results.png', dpi=150)
    plt.close()
    print("Results saved to 'temperature_forecast_results.png'")

def main():
    print("=" * 50)
    print("Day 24: Time Series Forecasting for Temperature")
    print("=" * 50)
    
    # Generate data
    print("\n[1] Generating temperature data...")
    df = generate_temperature_data(days=730)
    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
    
    # Split data
    train_size = len(df) - 30
    train_data = df['temperature'].values[:train_size]
    test_data = df['temperature'].values[train_size:]
    test_dates = df.index[train_size:]
    
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    forecasts = {}
    
    # Moving Average
    print("\n[2] Training Moving Average model...")
    ma_model = MovingAverageModel(window=7)
    ma_model.fit(train_data)
    forecasts['Moving Avg'] = ma_model.predict(30)
    
    # Exponential Smoothing
    print("[3] Training Exponential Smoothing...")
    es_model = ExponentialSmoothing(alpha=0.3)
    es_model.fit(train_data)
    forecasts['Exp Smooth'] = es_model.predict(30)
    
    # ARIMA
    print("[4] Training ARIMA model...")
    arima_model = SimpleARIMA(order=5)
    arima_model.fit(train_data)
    forecasts['ARIMA'] = arima_model.predict(30)
    
    # LSTM (if available)
    print("[5] Training LSTM model...")
    lstm_model, scaler = build_lstm_forecaster(train_data, lookback=30, epochs=30)
    if lstm_model:
        last_seq = train_data[-30:].reshape(-1, 1)
        scaled_seq = scaler.transform(last_seq).flatten()
        
        lstm_preds = []
        current_seq = list(scaled_seq)
        
        for _ in range(30):
            pred = lstm_model.predict(np.array([current_seq]).reshape(1, 30, 1), verbose=0)[0, 0]
            lstm_preds.append(pred)
            current_seq.pop(0)
            current_seq.append(pred)
        
        forecasts['LSTM'] = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()
    else:
        print("  LSTM not available (TensorFlow not installed)")
    
    # Evaluate models
    print("\n[6] Evaluating models...")
    print("\nModel Performance:")
    print("-" * 50)
    
    results = {}
    for name, pred in forecasts.items():
        metrics = evaluate_forecast(test_data, pred)
        results[name] = metrics
        print(f"{name:15} - RMSE: {metrics['RMSE']:.2f}°C, MAE: {metrics['MAE']:.2f}°C, MAPE: {metrics['MAPE']:.1f}%")
    
    # Best model
    best_model = min(results, key=lambda x: results[x]['RMSE'])
    print(f"\nBest Model: {best_model}")
    
    # Plot results
    print("\n[7] Generating visualizations...")
    plot_forecast_results(df, forecasts, test_data, test_dates)
    
    # Save results
    output = {
        'models': {name: {k: float(v) for k, v in metrics.items()} 
                   for name, metrics in results.items()},
        'best_model': best_model,
        'forecast_days': 30
    }
    
    with open('forecast_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Results saved to 'forecast_results.json'")
    
    print("\n" + "=" * 50)
    print("Day 24 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
