"""
Time Series Forecasting - Part 1: Statistical Foundations
=========================================================

This module covers:
- Time series data structures
- Visualization and decomposition
- Stationarity tests
- ACF and PACF analysis
- Basic forecasting methods

Based on: Financial Forecasting (Stock Prices)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. TIME SERIES DATA STRUCTURE
# ============================================================================

print("=" * 70)
print("1. TIME SERIES DATA STRUCTURE")
print("=" * 70)

# Create sample stock price data
np.random.seed(42)
n = 500  # Trading days (~2 years)

# Generate dates (business days)
dates = pd.date_range(start='2022-01-01', periods=n, freq='B')

# Generate synthetic stock price data with trend and seasonality
trend = np.linspace(100, 150, n)  # Upward trend
seasonality = 10 * np.sin(np.linspace(0, 8 * np.pi, n))  # Seasonal pattern
noise = np.random.randn(n) * 3  # Random noise

# Combine components
prices = trend + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Close': prices,
    'Volume': np.random.randint(1000000, 10000000, n)
})

# Set date as index
df.set_index('Date', inplace=True)

print("Time Series DataFrame:")
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nData types:\n{df.dtypes}")

# ============================================================================
# 2. TIME SERIES VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("2. TIME SERIES VISUALIZATION")
print("=" * 70)

# Basic line plot
print("\nVisualizing time series...")
print("Close price statistics:")
print(f"  Mean: {df['Close'].mean():.2f}")
print(f"  Std: {df['Close'].std():.2f}")
print(f"  Min: {df['Close'].min():.2f}")
print(f"  Max: {df['Close'].max():.2f}")

# Calculate returns (percentage change)
df['Returns'] = df['Close'].pct_change() * 100

print("\nReturns statistics:")
print(f"  Mean: {df['Returns'].mean():.4f}%")
print(f"  Std: {df['Returns'].std():.4f}%")

# ============================================================================
# 3. TIME SERIES DECOMPOSITION
# ============================================================================

print("\n" + "=" * 70)
print("3. TIME SERIES DECOMPOSITION")
print("=" * 70)

from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
# Using additive model
decomposition = seasonal_decompose(df['Close'], model='additive', period=20)

print("Time Series Components:")
print("  1. Trend: Long-term direction")
print("  2. Seasonal: Repeating pattern")
print("  3. Residual: Random noise")

# Extract components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print("\nTrend statistics:")
print(f"  Start: {trend.dropna().iloc[0]:.2f}")
print(f"  End: {trend.dropna().iloc[-1]:.2f}")

print("\nSeasonal amplitude:")
print(f"  Max: {seasonal.max():.2f}")
print(f"  Min: {seasonal.min():.2f}")

print("\nResidual statistics:")
print(f"  Mean: {residual.dropna().mean():.4f}")
print(f"  Std: {residual.dropna().std():.2f}")

# ============================================================================
# 4. ROLLING STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("4. ROLLING STATISTICS")
print("=" * 70)

# Calculate rolling mean and standard deviation
window = 20  # 20 trading days (~1 month)

df['Rolling_Mean'] = df['Close'].rolling(window=window).mean()
df['Rolling_Std'] = df['Close'].rolling(window=window).std()

print(f"Rolling window: {window} days")
print("\nRolling Mean (last 5 values):")
print(df['Rolling_Mean'].tail())

print("\nRolling Std (last 5 values):")
print(df['Rolling_Std'].tail())

# Calculate moving average crossovers
short_window = 10
long_window = 30

df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()

print(f"\nSimple Moving Averages:")
print(f"  Short: {short_window}-day SMA")
print(f"  Long: {long_window}-day SMA")

# ============================================================================
# 5. STATIONARITY TESTS
# ============================================================================

print("\n" + "=" * 70)
print("5. STATIONARITY TESTS")
print("=" * 70)

from statsmodels.tsa.stattools import adfuller, kpss

# ADF Test (Augmented Dickey-Fuller)
print("\n" + "-" * 50)
print("5.1 Augmented Dickey-Fuller Test")
print("-" * 50)

def adf_test(series, name=''):
    """Perform ADF test and print results."""
    result = adfuller(series.dropna(), autolag='AIC')

    print(f"\n{name} ADF Test:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Lags Used: {result[2]}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")

    if result[1] < 0.05:
        print(f"  Result: STATIONARY (p < 0.05)")
    else:
        print(f"  Result: NON-STATIONARY (p >= 0.05)")

    return result[1] < 0.05

# Test original series
is_stationary = adf_test(df['Close'], 'Original Series')

# Test returns (should be stationary)
is_returns_stationary = adf_test(df['Returns'].dropna(), 'Returns')

# Test differenced series
df['Diff'] = df['Close'].diff()
is_diff_stationary = adf_test(df['Diff'].dropna(), 'Differenced Series')

# KPSS Test
print("\n" + "-" * 50)
print("5.2 KPSS Test")
print("-" * 50)

def kpss_test(series, name=''):
    """Perform KPSS test."""
    # 'c' for level stationarity, 'ct' for trend stationarity
    result = kpss(series.dropna(), regression='c', nlags='auto')

    print(f"\n{name} KPSS Test:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Lags Used: {result[2]}")
    print(f"  Critical Values:")
    for key, value in result[3].items():
        print(f"    {key}: {value:.4f}")

    if result[1] > 0.05:
        print(f"  Result: STATIONARY (p > 0.05)")
    else:
        print(f"  Result: NON-STATIONARY (p <= 0.05)")

    return result[1] > 0.05

kpss_test(df['Close'], 'Original Series')

# ============================================================================
# 6. AUTOCORRELATION (ACF & PACF)
# ============================================================================

print("\n" + "=" * 70)
print("6. AUTOCORRELATION ANALYSIS")
print("=" * 70)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# Calculate ACF and PACF
nlags = 40

acf_values = acf(df['Close'].dropna(), nlags=nlags)
pacf_values = pacf(df['Close'].dropna(), nlags=nlags)

print("\nAutocorrelation Function (ACF):")
print(f"  Lag 1: {acf_values[1]:.4f}")
print(f"  Lag 5: {acf_values[5]:.4f}")
print(f"  Lag 10: {acf_values[10]:.4f}")
print(f"  Lag 20: {acf_values[20]:.4f}")

print("\nPartial Autocorrelation Function (PACF):")
print(f"  Lag 1: {pacf_values[1]:.4f}")
print(f"  Lag 5: {pacf_values[5]:.4f}")
print(f"  Lag 10: {pacf_values[10]:.4f}")
print(f"  Lag 20: {pacf_values[20]:.4f}")

# ============================================================================
# 7. BASIC FORECASTING METHODS
# ============================================================================

print("\n" + "=" * 70)
print("7. BASIC FORECASTING METHODS")
print("=" * 70)

# Split into train and test
train_size = int(len(df) * 0.8)
train = df['Close'][:train_size]
test = df['Close'][train_size:]

print(f"Training set: {len(train)} samples")
print(f"Test set: {len(test)} samples")

# -------------------------------------------------------------------------
# 7.1 Naive Method (Last Value)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("7.1 Naive Method")
print("-" * 50)

naive_forecast = np.full(len(test), train.iloc[-1])
naive_error = np.abs(test.values - naive_forecast)

print(f"Forecast: Use last value for all predictions")
print(f"MAE: {naive_error.mean():.4f}")
print(f"RMSE: {np.sqrt((naive_error**2).mean()):.4f}")

# -------------------------------------------------------------------------
# 7.2 Simple Average
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("7.2 Simple Average")
print("-" * 50)

avg_forecast = np.full(len(test), train.mean())
avg_error = np.abs(test.values - avg_forecast)

print(f"Forecast: Use mean of training data")
print(f"MAE: {avg_error.mean():.4f}")
print(f"RMSE: {np.sqrt((avg_error**2).mean()):.4f}")

# -------------------------------------------------------------------------
# 7.3 Moving Average
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("7.3 Moving Average")
print("-" * 50)

window = 20
ma_value = train.iloc[-window:].mean()
ma_forecast = np.full(len(test), ma_value)
ma_error = np.abs(test.values - ma_forecast)

print(f"Forecast: Use {window}-day moving average")
print(f"MAE: {ma_error.mean():.4f}")
print(f"RMSE: {np.sqrt((ma_error**2).mean()):.4f}")

# -------------------------------------------------------------------------
# 7.4 Exponential Smoothing (Simple)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("7.4 Simple Exponential Smoothing")
print("-" * 50)

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Fit model
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.3)
ses_forecast = ses_model.forecast(len(test))
ses_error = np.abs(test.values - ses_forecast.values)

print(f"Forecast: Exponential weighted moving average")
print(f"Alpha (smoothing): 0.3")
print(f"MAE: {ses_error.mean():.4f}")
print(f"RMSE: {np.sqrt((ses_error**2).mean()):.4f}")

# ============================================================================
# 8. LAG FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("8. LAG FEATURES FOR ML MODELS")
print("=" * 70)

# Create lag features
df_lagged = pd.DataFrame()

for lag in [1, 2, 3, 5, 10, 20]:
    df_lagged[f'lag_{lag}'] = df['Close'].shift(lag)

# Add rolling features
df_lagged['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
df_lagged['rolling_std_5'] = df['Close'].rolling(window=5).std()
df_lagged['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
df_lagged['rolling_std_20'] = df['Close'].rolling(window=20).std()

# Add target
df_lagged['target'] = df['Close']

# Drop NaN rows
df_lagged = df_lagged.dropna()

print("Lag Features created:")
print(f"  Features: {list(df_lagged.columns)}")
print(f"  Shape: {df_lagged.shape}")
print("\nSample:")
print(df_lagged.head())

# ============================================================================
# 9. TIME SERIES WITH PANDAS
# ============================================================================

print("\n" + "=" * 70)
print("9. PANDAS TIME SERIES OPERATIONS")
print("=" * 70)

# Resampling
print("\n9.1 Resampling (Daily to Monthly)")
print("-" * 50)

# Create daily data
daily_dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
daily_data = pd.Series(np.random.randn(365), index=daily_dates)

# Resample to monthly
monthly = daily_data.resample('M').mean()
print(f"Daily data: {len(daily_data)} points")
print(f"Monthly data: {len(monthly)} points")

# Other resample frequencies
print("\nAvailable frequencies:")
print("  D: Daily")
print("  W: Weekly")
print("  M: Monthly")
print("  Q: Quarterly")
print("  Y: Yearly")

# Shifting
print("\n9.2 Shifting")
print("-" * 50)

shifted = df['Close'].shift(1)
print(f"Original: {df['Close'].iloc[0]:.2f}")
print(f"Shifted +1: {shifted.iloc[1]:.2f}")

# Date-time indexing
print("\n9.3 Date-time Indexing")
print("-" * 50)

# Select date range
subset = df['Close']['2022-06-01':'2022-12-31']
print(f"Date range selection (Jun-Dec 2022): {len(subset)} points")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Time Series Structure
   - Index: Date/time values
   - Values: The metric we're forecasting

2. Decomposition
   - Trend: Long-term movement
   - Seasonality: Repeating patterns
   - Residual: Random noise

3. Stationarity
   - ADF: Tests for unit root (null: non-stationary)
   - KPSS: Tests stationarity (null: stationary)
   - Differencing can make series stationary

4. ACF/PACF
   - ACF: Correlation with lagged values
   - PACF: Direct correlation (excluding indirect)
   - Used for choosing ARIMA parameters

5. Basic Forecasting
   - Naive: Last value
   - Simple Average: Mean of all history
   - Moving Average: Mean of recent values
   - Exponential Smoothing: Weighted average

6. Feature Engineering
   - Lag features: Past values
   - Rolling statistics: Moving averages, std
   - Date features: Day of week, month, etc.

Next: ARIMA Models (02_arima_models.py)
""")
