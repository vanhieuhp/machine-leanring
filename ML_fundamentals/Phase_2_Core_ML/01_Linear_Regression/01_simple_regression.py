"""
Linear Regression - Part 1: Simple Regression
==============================================

This module covers:
- Simple linear regression (one feature)
- Fitting a line to data
- Making predictions
- Evaluating the model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================================
# 1. SIMPLE LINEAR REGRESSION FROM SCRATCH
# ============================================================================

print("=" * 70)
print("1. SIMPLE LINEAR REGRESSION FROM SCRATCH")
print("=" * 70)

# Generate sample data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 2 * X + 1 + np.random.randn(10) * 2  # y = 2x + 1 + noise

print(f"X: {X}")
print(f"y: {y}")

# Calculate coefficients manually
n = len(X)
mean_x = np.mean(X)
mean_y = np.mean(y)

# Slope (m)
numerator = np.sum((X - mean_x) * (y - mean_y))
denominator = np.sum((X - mean_x) ** 2)
m = numerator / denominator

# Intercept (b)
b = mean_y - m * mean_x

print(f"\nCalculated coefficients:")
print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Equation: y = {m:.4f}x + {b:.4f}")

# Make predictions
y_pred = m * X + b
print(f"\nPredictions: {y_pred}")

# ============================================================================
# 2. EVALUATION METRICS
# ============================================================================

print("\n" + "=" * 70)
print("2. EVALUATION METRICS")
print("=" * 70)

# Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)
print(f"MSE: {mse:.4f}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(y - y_pred))
print(f"MAE: {mae:.4f}")

# R² Score
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - mean_y) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"R² Score: {r2:.4f}")

# ============================================================================
# 3. USING SCIKIT-LEARN
# ============================================================================

print("\n" + "=" * 70)
print("3. USING SCIKIT-LEARN")
print("=" * 70)

# Reshape data for sklearn
X_train = X.reshape(-1, 1)

# Create and train model
model = LinearRegression()
model.fit(X_train, y)

# Get coefficients
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Make predictions
y_pred_sklearn = model.predict(X_train)

# Evaluate
mse_sklearn = mean_squared_error(y, y_pred_sklearn)
rmse_sklearn = np.sqrt(mse_sklearn)
mae_sklearn = mean_absolute_error(y, y_pred_sklearn)
r2_sklearn = r2_score(y, y_pred_sklearn)

print(f"\nMetrics:")
print(f"MSE: {mse_sklearn:.4f}")
print(f"RMSE: {rmse_sklearn:.4f}")
print(f"MAE: {mae_sklearn:.4f}")
print(f"R² Score: {r2_sklearn:.4f}")

# ============================================================================
# 4. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("4. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data and fitted line
axes[0].scatter(X, y, color='blue', s=100, alpha=0.6, label='Actual data')
axes[0].plot(X, y_pred_sklearn, color='red', linewidth=2, label='Fitted line')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title(f'Linear Regression (R² = {r2_sklearn:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y - y_pred_sklearn
axes[1].scatter(X, residuals, color='green', s=100, alpha=0.6)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 5. PRACTICAL EXAMPLE: House Prices
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL EXAMPLE: House Prices")
print("=" * 70)

# Sample data: house size (sq ft) vs price
house_sizes = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
prices = np.array([200000, 280000, 350000, 420000, 500000, 580000, 650000, 720000, 800000])

print(f"House sizes: {house_sizes}")
print(f"Prices: {prices}")

# Train model
X_houses = house_sizes.reshape(-1, 1)
model_houses = LinearRegression()
model_houses.fit(X_houses, prices)

# Predictions
y_pred_houses = model_houses.predict(X_houses)

# Evaluate
r2_houses = r2_score(prices, y_pred_houses)
rmse_houses = np.sqrt(mean_squared_error(prices, y_pred_houses))

print(f"\nModel:")
print(f"Price = {model_houses.coef_[0]:.2f} * Size + {model_houses.intercept_:.2f}")
print(f"R² Score: {r2_houses:.4f}")
print(f"RMSE: ${rmse_houses:.2f}")

# Predict for new house
new_size = 3500
predicted_price = model_houses.predict([[new_size]])[0]
print(f"\nPredicted price for {new_size} sq ft house: ${predicted_price:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(house_sizes, prices, color='blue', s=100, alpha=0.6, label='Actual prices')
plt.plot(house_sizes, y_pred_houses, color='red', linewidth=2, label='Fitted line')
plt.scatter([new_size], [predicted_price], color='green', s=200, marker='*', label='Prediction')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title(f'House Price Prediction (R² = {r2_houses:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 6. MAKING PREDICTIONS ON NEW DATA
# ============================================================================

print("\n" + "=" * 70)
print("6. MAKING PREDICTIONS ON NEW DATA")
print("=" * 70)

# New data points
X_new = np.array([11, 12, 13, 14, 15]).reshape(-1, 1)
y_new_pred = model.predict(X_new)

print("New predictions:")
for x_val, y_val in zip(X_new.flatten(), y_new_pred):
    print(f"X = {x_val}: y = {y_val:.4f}")
