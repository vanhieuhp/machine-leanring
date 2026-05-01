"""
Time Series Forecasting - Practice Exercises
==========================================

Complete these exercises to solidify your time series skills.
Solutions are provided at the bottom.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXERCISE 1: Data Exploration
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Data Exploration")
print("=" * 70)

# 1.1 Create sample time series data
# TODO: Create a DataFrame with date index and random values
np.random.seed(42)

# Create dates (daily for 1 year)
dates = None  # TODO: Create date range

# Create values with trend and seasonality
values = None  # TODO: Create values with trend + seasonality + noise

# Create DataFrame
df = None  # TODO: Create DataFrame

# 1.2 Calculate basic statistics
# TODO: Calculate mean, std, min, max
mean_val = None
std_val = None

print(f"Mean: {mean_val}")
print(f"Std: {std_val}")

# 1.3 Calculate rolling statistics
# TODO: Calculate 7-day and 30-day rolling mean
rolling_7 = None  # TODO: Calculate
rolling_30 = None  # TODO: Calculate

print("\nRolling statistics (last 5):")
print(rolling_30.tail())

# ============================================================================
# EXERCISE 2: Stationarity
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Stationarity Tests")
print("=" * 70)

# 2.1 ADF: Perform Test
# TODO ADF test on the data
result = None  # TODO: adfuller()

print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")

if result[1] < 0.05:
    print("Result: Stationary")
else:
    print("Result: Non-stationary")

# 2.2 Make series stationary
# TODO: Difference the series
diff_series = None  # TODO: .diff().dropna()

# Test again
result_diff = None  # TODO: adfuller()
print(f"\nAfter differencing p-value: {result_diff[1]:.4f}")

# ============================================================================
# EXERCISE 3: ACF and PACF
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: ACF and PACF")
print("=" * 70)

# 3.1 Calculate ACF
# TODO: Calculate ACF with 20 lags
acf_values = None  # TODO: acf()

print("ACF values:")
for i in range(1, 11):
    print(f"  Lag {i}: {acf_values[i]:.4f}")

# 3.2 Calculate PACF
# TODO: Calculate PACF with 20 lags
pacf_values = None  # TODO: pacf()

print("\nPACF values:")
for i in range(1, 11):
    print(f"  Lag {i}: {pacf_values[i]:.4f}")

# ============================================================================
# EXERCISE 4: Decomposition
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Time Series Decomposition")
print("=" * 70)

# 4.1 Decompose the series
# TODO: Use seasonal_decompose
decomposition = None  # TODO: seasonal_decompose()

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print(f"Trend range: {trend.min():.2f} to {trend.max():.2f}")
print(f"Seasonal amplitude: {seasonal.max() - seasonal.min():.2f}")
print(f"Residual std: {residual.std():.2f}")

# ============================================================================
# EXERCISE 5: ARIMA
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: ARIMA Modeling")
print("=" * 70)

# Create sample data
np.random.seed(42)
data = pd.Series(np.cumsum(np.random.randn(100)) + 100)

# Split
train = data[:80]
test = data[80:]

# 5.1 Fit ARIMA model
# TODO: Fit ARIMA(1,1,1)
model = None  # TODO: ARIMA()
fitted = None  # TODO: .fit()

# 5.2 Forecast
# TODO: Forecast 20 steps
forecast = None  # TODO: .forecast()

# 5.3 Calculate error
mae = None  # TODO: mean_absolute_error()
rmse = None  # TODO: np.sqrt(mean_squared_error())

print(f"Forecast MAE: {mae:.4f}")
print(f"Forecast RMSE: {rmse:.4f}")

# ============================================================================
# EXERCISE 6: Grid Search ARIMA
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Grid Search ARIMA")
print("=" * 70)

# 6.1 Try different parameters
# TODO: Try (1,1,1), (2,1,1), (1,1,2), (2,1,2)

best_aic = float('inf')
best_order = None

for p in range(3):
    for d in range(2):
        for q in range(3):
            try:
                model = ARIMA(train, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue

print(f"Best order: {best_order}")
print(f"Best AIC: {best_aic:.2f}")

# ============================================================================
# EXERCISE 7: Exponential Smoothing
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Exponential Smoothing")
print("=" * 70)

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# 7.1 Simple Exponential Smoothing
# TODO: Fit SES
ses_model = None  # TODO: SimpleExpSmoothing()
ses_fit = None  # TODO: .fit()
ses_forecast = None  # TODO: .forecast()

# 7.2 Holt's Linear Trend
# TODO: Fit Holt
holt_model = None  # TODO: Holt()
holt_fit = None  # TODO: .fit()
holt_forecast = None  # TODO: .forecast()

# 7.3 Compare
ses_mae = mean_absolute_error(test, ses_forecast)
holt_mae = mean_absolute_error(test, holt_forecast)

print(f"SES MAE: {ses_mae:.4f}")
print(f"Holt MAE: {holt_mae:.4f}")

# ============================================================================
# EXERCISE 8: Deep Learning Data Preparation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Deep Learning Data Preparation")
print("=" * 70)

from sklearn.preprocessing import MinMaxScaler

# Create data
np.random.seed(42)
data = np.cumsum(np.random.randn(100)) + 100

# 8.1 Scale data
# TODO: Scale data to 0-1
scaler = None  # TODO: MinMaxScaler()
scaled_data = None  # TODO: scaler.fit_transform()

# 8.2 Create sequences
# TODO: Create sequences with window size 10
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = None  # TODO: create_sequences()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 8.3 Reshape for LSTM [samples, timesteps, features]
X = None  # TODO: X.reshape()
print(f"Reshaped X: {X.shape}")

# ============================================================================
# EXERCISE 9: Feature Engineering for Time Series
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9: Feature Engineering")
print("=" * 70)

# Create sample data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
df = pd.DataFrame({'value': np.random.randn(100)}, index=dates)

# 9.1 Date features
# TODO: Extract date features
df['dayofweek'] = None  # TODO: .dayofweek
df['month'] = None  # TODO: .month
df['quarter'] = None  # TODO: .quarter
df['dayofyear'] = None  # TODO: .dayofyear

# 9.2 Lag features
# TODO: Create lag features
for lag in [1, 2, 3, 7, 14]:
    df[f'lag_{lag}'] = None  # TODO: shift()

# 9.3 Rolling features
# TODO: Create rolling features
df['rolling_mean_7'] = None  # TODO: rolling().mean()
df['rolling_std_7'] = None  # TODO: rolling().std()

print("Features created:")
print(df.columns.tolist())

# ============================================================================
# EXERCISE 10: Complete Forecasting Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 10: Complete Pipeline")
print("=" * 70)

def forecast_pipeline(data, test_size=0.2):
    """
    Complete time series forecasting pipeline.

    Steps:
    1. Explore data
    2. Check stationarity
    3. Decompose
    4. Fit ARIMA
    5. Evaluate
    """
    # Split
    split = int(len(data) * (1 - test_size))
    train = data[:split]
    test = data[split:]

    # 1. Stationarity
    # TODO: Check stationarity
    result = None  # TODO: adfuller()
    d = 1 if result[1] > 0.05 else 0

    # 2. Fit ARIMA
    # TODO: Fit ARIMA
    model = None  # TODO: ARIMA
    fitted = None  # TODO: .fit()

    # 3. Forecast
    # TODO: Forecast
    forecast = None  # TODO: .forecast()

    # 4. Evaluate
    # TODO: Calculate metrics
    mae = None  # TODO: MAE
    rmse = None  # TODO: RMSE

    return {'mae': mae, 'rmse': rmse, 'forecast': forecast}

# Test pipeline
np.random.seed(42)
test_data = pd.Series(np.cumsum(np.random.randn(100)) + 100)
# results = forecast_pipeline(test_data)
# print(f"Results: {results}")

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("""
EXERCISE 1:
1.1: dates = pd.date_range(start='2023-01-01', periods=365)
      values = np.linspace(100, 200, 365) + np.sin(np.linspace(0, 10*np.pi, 365)) * 10 + np.random.randn(365) * 5
      df = pd.DataFrame({'value': values}, index=dates)

1.2: mean_val = df['value'].mean()
      std_val = df['value'].std()

1.3: rolling_7 = df['value'].rolling(7).mean()
      rolling_30 = df['value'].rolling(30).mean()

EXERCISE 2:
2.1: result = adfuller(df['value'])
2.2: diff_series = df['value'].diff().dropna()

EXERCISE 3:
3.1: acf_values = acf(series, nlags=20)
3.2: pacf_values = pacf(series, nlags=20)

EXERCISE 4:
4.1: decomposition = seasonal_decompose(df['value'], model='additive', period=30)

EXERCISE 5:
5.1: model = ARIMA(train, order=(1,1,1)); fitted = model.fit()
5.2: forecast = fitted.forecast(steps=20)
5.3: mae = mean_absolute_error(test, forecast); rmse = np.sqrt(mean_squared_error(test, forecast))

EXERCISE 6:
6.1: Grid search over p, d, q ranges with ARIMA

EXERCISE 7:
7.1: ses_model = SimpleExpSmoothing(train).fit(); ses_forecast = ses_model.forecast(len(test))
7.2: holt_model = Holt(train).fit(); holt_forecast = holt_model.forecast(len(test))

EXERCISE 8:
8.1: scaler = MinMaxScaler(); scaled_data = scaler.fit_transform(data.reshape(-1,1))
8.2: X, y = create_sequences(scaled_data.flatten(), 10)
8.3: X = X.reshape((X.shape[0], X.shape[1], 1))

EXERCISE 9:
9.1: df['dayofweek'] = df.index.dayofweek; etc.
9.2: df['lag_1'] = df['value'].shift(1)
9.3: df['rolling_mean_7'] = df['value'].rolling(7).mean()

EXERCISE 10:
10.1: Combine all steps in forecast_pipeline function
""")
