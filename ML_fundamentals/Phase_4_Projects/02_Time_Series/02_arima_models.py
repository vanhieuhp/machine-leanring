"""
Time Series Forecasting - Part 2: ARIMA Models
===============================================

This module covers:
- ARIMA fundamentals
- Parameter selection (p, d, q)
- SARIMA for seasonal data
- Model diagnostics
- Forecasting

Based on: Financial Forecasting (Stock Prices)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CREATE SAMPLE DATA
# ============================================================================

print("=" * 70)
print("1. PREPARING SAMPLE DATA")
print("=" * 70)

np.random.seed(42)
n = 500

# Generate dates
dates = pd.date_range(start='2022-01-01', periods=n, freq='B')

# Generate stock-like data with trend, seasonality, and AR component
trend = np.linspace(100, 150, n)
seasonal = 10 * np.sin(np.linspace(0, 8 * np.pi, n))

# AR(1) component
ar_component = np.zeros(n)
ar_component[0] = np.random.randn()
for i in range(1, n):
    ar_component[i] = 0.7 * ar_component[i-1] + np.random.randn()

noise = np.random.randn(n) * 3

prices = trend + seasonal + ar_component * 5 + noise

df = pd.DataFrame({
    'Date': dates,
    'Close': prices
})
df.set_index('Date', inplace=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nSample data:")
print(df.head())

# Split data
train_size = int(len(df) * 0.8)
train = df['Close'][:train_size]
test = df['Close'][train_size:]

print(f"\nTrain: {len(train)} samples")
print(f"Test: {len(test)} samples")

# ============================================================================
# 2. ARIMA FUNDAMENTALS
# ============================================================================

print("\n" + "=" * 70)
print("2. ARIMA FUNDAMENTALS")
print("=" * 70)

print("""
ARIMA Components:
-----------------
AR (p): Autoregressive - uses past values
   - p: Number of lag observations

I (d): Integrated - differencing to make stationary
   - d: Number of differencing required

MA (q): Moving Average - uses past errors
   - q: Size of moving average window

Order: (p, d, q)
""")

# -------------------------------------------------------------------------
# 2.1 Determine Differencing Order (d)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.1 Determining Differencing Order (d)")
print("-" * 50)

def find_d(series, max_d=3):
    """Find the differencing order."""
    for d in range(max_d + 1):
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()

        result = adfuller(test_series, autolag='AIC')
        print(f"d={d}: ADF p-value = {result[1]:.4f}")

        if result[1] < 0.05:
            print(f"  → Stationary! d={d}")
            return d

    print(f"  → Using d={max_d}")
    return max_d

d_order = find_d(train)
print(f"\nOptimal differencing order: d={d_order}")

# -------------------------------------------------------------------------
# 2.2 Determine AR Order (p) using PACF
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.2 Determining AR Order (p)")
print("-" * 50)

# After differencing
diff_series = train.diff(d_order).dropna()

# PACF of differenced series
pacf_values = pacf(diff_series, nlags=20)

print("PACF of differenced series:")
for i in range(1, 11):
    sig = "***" if abs(pacf_values[i]) > 1.96/np.sqrt(len(diff_series)) else ""
    print(f"  Lag {i}: {pacf_values[i]:.4f} {sig}")

# Find significant lags
significant_p = 0
for i in range(1, len(pacf_values)):
    if abs(pacf_values[i]) > 1.96/np.sqrt(len(diff_series)):
        significant_p = i

print(f"\nSuggested p (AR order): {significant_p}")

# -------------------------------------------------------------------------
# 2.3 Determine MA Order (q) using ACF
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.3 Determining MA Order (q)")
print("-" * 50)

# ACF of differenced series
acf_values = acf(diff_series, nlags=20)

print("ACF of differenced series:")
for i in range(1, 11):
    sig = "***" if abs(acf_values[i]) > 1.96/np.sqrt(len(diff_series)) else ""
    print(f"  Lag {i}: {acf_values[i]:.4f} {sig}")

# Find significant lags
significant_q = 0
for i in range(1, len(acf_values)):
    if abs(acf_values[i]) > 1.96/np.sqrt(len(diff_series)):
        significant_q = i

print(f"\nSuggested q (MA order): {significant_q}")

# ============================================================================
# 3. FITTING ARIMA MODEL
# ============================================================================

print("\n" + "=" * 70)
print("3. FITTING ARIMA MODEL")
print("=" * 70)

# Use suggested parameters
p, d, q = significant_p, d_order, significant_q
print(f"Fitting ARIMA({p}, {d}, {q})...")

# Fit ARIMA model
model = ARIMA(train, order=(p, d, q))
fitted_model = model.fit()

print("\nModel Summary:")
print(f"  AIC: {fitted_model.aic:.2f}")
print(f"  BIC: {fitted_model.bic:.2f}")
print(f"  Log Likelihood: {fitted_model.llf:.2f}")

print("\nParameters:")
print(fitted_model.summary().tables[1])

# ============================================================================
# 4. ARIMA FORECASTING
# ============================================================================

print("\n" + "=" * 70)
print("4. ARIMA FORECASTING")
print("=" * 70)

# Make predictions
n_forecast = len(test)
forecast = fitted_model.forecast(steps=n_forecast)

# Calculate errors
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100

print(f"\nARIMA({p},{d},{q}) Performance:")
print(f"  MAE: {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAPE: {mape:.2f}%")

# -------------------------------------------------------------------------
# 4.1 Rolling Forecast
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.1 Rolling Forecast (One Step Ahead)")
print("-" * 50)

# Rolling forecast
rolling_predictions = []
history = train.tolist()

for i in range(len(test)):
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    rolling_predictions.append(yhat)
    history.append(test.iloc[i])

rolling_mae = mean_absolute_error(test, rolling_predictions)
rolling_rmse = np.sqrt(mean_squared_error(test, rolling_predictions))

print(f"Rolling Forecast Performance:")
print(f"  MAE: {rolling_mae:.4f}")
print(f"  RMSE: {rolling_rmse:.4f}")

# ============================================================================
# 5. GRID SEARCH FOR ARIMA
# ============================================================================

print("\n" + "=" * 70)
print("5. GRID SEARCH FOR ARIMA PARAMETERS")
print("=" * 70)

# Define parameter ranges
p_range = range(0, 4)
d_range = range(0, 3)
q_range = range(0, 4)

best_aic = float('inf')
best_order = None
results = []

print("Searching for best parameters...")
print("(This may take a while...)")

# Note: In practice, use smaller ranges for speed
for p in p_range[:2]:  # Reduce for demo
    for d in d_range[:2]:
        for q in q_range[:2]:
            try:
                model = ARIMA(train, order=(p, d, q))
                fitted = model.fit()
                aic = fitted.aic
                results.append({'order': (p, d, q), 'aic': aic})

                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)

            except:
                continue

print(f"\nBest order: {best_order}")
print(f"Best AIC: {best_aic:.2f}")

# Show top results
results_df = pd.DataFrame(results).sort_values('aic')
print("\nTop 5 models:")
print(results_df.head())

# Fit best model
best_model = ARIMA(train, order=best_order)
best_fitted = best_model.fit()

best_forecast = best_fitted.forecast(steps=n_forecast)
best_mae = mean_absolute_error(test, best_forecast)

print(f"\nBest model forecast MAE: {best_mae:.4f}")

# ============================================================================
# 6. SARIMA (SEASONAL ARIMA)
# ============================================================================

print("\n" + "=" * 70)
print("6. SARIMA (SEASONAL ARIMA)")
print("=" * 70)

print("""
SARIMA extends ARIMA with seasonal components:

Order: (p, d, q) x (P, D, Q, s)

Non-seasonal:
  p: AR order
  d: Differencing
  q: MA order

Seasonal:
  P: Seasonal AR order
  D: Seasonal differencing
  Q: Seasonal MA order
  s: Seasonal period (e.g., 12 for monthly, 7 for daily)
""")

# Create data with clear seasonality
np.random.seed(42)
n = 500

# Weekly seasonality (period = 5 for business days)
seasonal_period = 5

dates = pd.date_range(start='2022-01-01', periods=n, freq='B')
trend = np.linspace(100, 150, n)
seasonal = 15 * np.sin(np.linspace(0, 2 * np.pi * n / seasonal_period, n))
noise = np.random.randn(n) * 3

prices = trend + seasonal + noise

df_seasonal = pd.DataFrame({'Close': prices}, index=dates)

# Split
train_seasonal = df_seasonal['Close'][:int(0.8*n)]
test_seasonal = df_seasonal['Close'][int(0.8*n):]

# Fit SARIMA
# Order: (p,d,q) x (P,D,Q,s)
seasonal_order = (1, 1, 1, 5)  # Weekly seasonality

print(f"\nFitting SARIMA{seasonal_order}...")

sarima_model = SARIMAX(
    train_seasonal,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 5)
)
sarima_fit = sarima_model.fit(disp=False)

print(f"\nSARIMA Summary:")
print(f"  AIC: {sarima_fit.aic:.2f}")
print(f"  BIC: {sarima_fit.bic:.2f}")

# Forecast
sarima_forecast = sarima_fit.forecast(steps=len(test_seasonal))
sarima_mae = mean_absolute_error(test_seasonal, sarima_forecast)

print(f"\nSARIMA Performance:")
print(f"  MAE: {sarima_mae:.4f}")

# Compare with non-seasonal ARIMA
arima_model = ARIMA(train_seasonal, order=(1, 1, 1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test_seasonal))
arima_mae = mean_absolute_error(test_seasonal, arima_forecast)

print(f"\nARIMA (non-seasonal) Performance:")
print(f"  MAE: {arima_mae:.4f}")
print(f"\nImprovement: {(arima_mae - sarima_mae) / arima_mae * 100:.1f}%")

# ============================================================================
# 7. MODEL DIAGNOSTICS
# ============================================================================

print("\n" + "=" * 70)
print("7. MODEL DIAGNOSTICS")
print("=" * 70)

# Get residuals
residuals = best_fitted.resid

print("\nResiduals Statistics:")
print(f"  Mean: {residuals.mean():.4f}")
print(f"  Std: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("\nLjung-Box Test:")
print(f"  p-value: {lb_test['lb_pvalue'].values[0]:.4f}")
if lb_test['lb_pvalue'].values[0] > 0.05:
    print("  → Residuals are white noise (good)")
else:
    print("  → Residuals show autocorrelation (consider more lags)")

# Jarque-Bera test for normality
from scipy.stats import jarque_bera

jb_stat, jb_pvalue = jarque_bera(residuals)
print("\nJarque-Bera Test:")
print(f"  Statistic: {jb_stat:.4f}")
print(f"  p-value: {jb_pvalue:.4f}")
if jb_pvalue > 0.05:
    print("  → Residuals are normally distributed (good)")
else:
    print("  → Residuals may not be normal")

# ============================================================================
# 8. EXPONENTIAL SMOOTHING METHODS
# ============================================================================

print("\n" + "=" * 70)
print("8. EXPONENTIAL SMOOTHING METHODS")
print("=" * 70)

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# -------------------------------------------------------------------------
# 8.1 Simple Exponential Smoothing
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("8.1 Simple Exponential Smoothing (SES)")
print("-" * 50)

ses_model = SimpleExpSmoothing(train).fit()
ses_forecast = ses_model.forecast(n_forecast)
ses_mae = mean_absolute_error(test, ses_forecast)

print(f"SES MAE: {ses_mae:.4f}")

# -------------------------------------------------------------------------
# 8.2 Holt's Linear Trend
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("8.2 Holt's Linear Trend")
print("-" * 50)

from statsmodels.tsa.holtwinters import Holt

holt_model = Holt(train, damped_trend=True).fit()
holt_forecast = holt_model.forecast(n_forecast)
holt_mae = mean_absolute_error(test, holt_forecast)

print(f"Holt's Linear MAE: {holt_mae:.4f}")

# -------------------------------------------------------------------------
# 8.3 Holt-Winters
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("8.3 Holt-Winters (Additive)")
print("-" * 50)

# Use seasonal period
hw_model = ExponentialSmoothing(
    train,
    seasonal_period=20,
    trend='add',
    seasonal='add'
).fit()

hw_forecast = hw_model.forecast(n_forecast)
hw_mae = mean_absolute_error(test, hw_forecast)

print(f"Holt-Winters MAE: {hw_mae:.4f}")

# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("9. MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': ['Naive', 'SES', 'Holt', 'Holt-Winters', 'ARIMA', 'SARIMA'],
    'MAE': [
        mean_absolute_error(test, np.full(len(test), train.iloc[-1])),
        ses_mae,
        holt_mae,
        hw_mae,
        best_mae,
        sarima_mae
    ]
})

print("\nModel Performance (MAE):")
comparison = comparison.sort_values('MAE')
print(comparison.to_string(index=False))

print(f"\nBest Model: {comparison.iloc[0]['Model']} (MAE: {comparison.iloc[0]['MAE']:.4f})")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. ARIMA Parameters
   - p (AR): Use PACF to find
   - d (I): Make series stationary
   - q (MA): Use ACF to find

2. Model Selection
   - Use AIC/BIC for model comparison
   - Lower is better
   - Grid search for optimal parameters

3. SARIMA
   - Add seasonal components
   - Useful for periodic data
   - s = seasonal period

4. Exponential Smoothing
   - SES: No trend/seasonality
   - Holt: Linear trend
   - Holt-Winters: Trend + seasonality

5. Diagnostics
   - Check residuals for patterns
   - Ljung-Box: Test for autocorrelation
   - Normality: Jarque-Bera test

6. Forecasting Methods
   - Static: Train once, forecast all
   - Rolling: Retrain for each step (more accurate)

Next: Deep Learning for Time Series (03_deep_learning.py)
""")
