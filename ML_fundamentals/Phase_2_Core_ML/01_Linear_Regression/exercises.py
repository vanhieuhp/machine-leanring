"""
Linear Regression - Exercises
==============================

Practice problems for Linear Regression.
Solutions are provided at the bottom.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# EXERCISE 1: Simple Linear Regression
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Simple Linear Regression")
print("=" * 70)

# 1.1 Create sample data: X (hours studied) vs y (test score)
# Hours: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Scores: [35, 42, 50, 55, 62, 68, 72, 78, 82, 88]

# TODO: Create the data
hours = None
scores = None

# 1.2 Fit a linear regression model
# TODO: Fit model

# 1.3 Predict score for 11 hours of studying
# TODO: Predict

# 1.4 Calculate R² score
# TODO: Calculate

# 1.5 Calculate RMSE
# TODO: Calculate

# ============================================================================
# EXERCISE 2: Multiple Linear Regression
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Multiple Linear Regression")
print("=" * 70)

# 2.1 Create dataset with multiple features
# Features: study_hours, sleep_hours, attendance (%)
# Target: final_exam_score

data = {
    'study_hours': [5, 3, 7, 2, 8, 4, 6, 1, 9, 5],
    'sleep_hours': [7, 6, 6, 5, 7, 6, 7, 4, 8, 6],
    'attendance': [90, 80, 95, 70, 100, 85, 92, 65, 98, 88],
    'score': [75, 60, 85, 50, 92, 68, 80, 45, 95, 72]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# 2.2 Separate features and target
# TODO: Separate X and y

# 2.3 Split into train/test (80/20)
# TODO: Split

# 2.4 Train linear regression
# TODO: Train

# 2.5 Evaluate on test set
# TODO: Evaluate

# 2.6 Which feature has highest coefficient?
# TODO: Check coefficients

# ============================================================================
# EXERCISE 3: Gradient Descent
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Gradient Descent Implementation")
print("=" * 70)

# 3.1 Implement simple gradient descent for linear regression

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Implement gradient descent for linear regression.

    Args:
        X: Features (normalized)
        y: Target
        learning_rate: Step size
        iterations: Number of iterations

    Returns:
        m, b: Coefficients
    """
    m = 0
    b = 0
    n = len(X)

    # TODO: Implement gradient descent
    # Hint: Calculate predictions, errors, gradients, update m and b

    return m, b

# Test with sample data
X_test = np.array([1, 2, 3, 4, 5])
y_test = np.array([2, 4, 5, 4, 5])

# Normalize for gradient descent
X_norm = (X_test - np.mean(X_test)) / np.std(X_test)

m_gd, b_gd = gradient_descent(X_norm, y_test)
print(f"Gradient descent result: m={m_gd:.4f}, b={b_gd:.4f}")

# 3.2 Compare with sklearn
model_sklearn = LinearRegression()
model_sklearn.fit(X_test.reshape(-1, 1), y_test)
print(f"sklearn result: m={model_sklearn.coef_[0]:.4f}, b={model_sklearn.intercept_:.4f}")

# ============================================================================
# EXERCISE 4: Feature Scaling
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Feature Scaling")
print("=" * 70)

# 4.1 Create data with different scales
np.random.seed(42)
X1 = np.random.rand(100) * 1000  # 0-1000
X2 = np.random.rand(100) * 10    # 0-10
X3 = np.random.rand(100)         # 0-1

# Target: depends on all
y = 0.5 * X1 + 10 * X2 + 100 * X3 + np.random.randn(100) * 10

# 4.2 Train WITHOUT scaling
X_no_scale = np.column_stack([X1, X2, X3])
model_no_scale = LinearRegression()
model_no_scale.fit(X_no_scale, y)

print("Without scaling:")
print(f"  Coefficients: {model_no_scale.coef_}")

# 4.3 Train WITH scaling
# TODO: Scale features and train

# 4.4 Compare coefficients - which is more meaningful?
# TODO: Compare

# ============================================================================
# EXERCISE 5: Polynomial Regression
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Polynomial Regression")
print("=" * 70)

# 5.1 Generate non-linear data
np.random.seed(42)
X_poly = np.linspace(0, 10, 50)
y_poly = X_poly**2 + np.random.randn(50) * 10  # Quadratic relationship

# 5.2 Fit linear regression - what happens?
# TODO: Fit linear

# 5.3 Create polynomial features (degree 2)
# Hint: Use np.column_stack or create manually
# TODO: Create polynomial features

# 5.4 Fit polynomial regression
# TODO: Fit

# 5.5 Compare R² scores
# TODO: Compare

# ============================================================================
# EXERCISE 6: Real-world Problem - House Price Prediction
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Real-world Problem - House Price Prediction")
print("=" * 70)

# Dataset: House prices based on size, bedrooms, age
houses = pd.DataFrame({
    'size_sqft': [1400, 1600, 1700, 1875, 1100, 1550, 2100, 1750, 1600, 1450],
    'bedrooms': [3, 3, 2, 4, 2, 3, 4, 3, 3, 3],
    'age_years': [10, 15, 5, 8, 20, 12, 3, 7, 10, 15],
    'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
})

print("House Prices Dataset:")
print(houses.head())

# 6.1 Split features and target
# TODO: X, y

# 6.2 Split train/test (80/20)
# TODO: X_train, X_test, y_train, y_test

# 6.3 Train model
# TODO: Train

# 6.4 Predict for a new house:
# - 2000 sqft, 4 bedrooms, 5 years old
# TODO: Predict

# 6.5 Calculate metrics
# TODO: RMSE, R²

# 6.6 Which factor affects price most?
# TODO: Analyze coefficients

# ============================================================================
# EXERCISE 7: Regularization (Ridge)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Regularization")
print("=" * 70)

from sklearn.linear_model import Ridge

# 7.1 Create data with many correlated features
np.random.seed(42)
X_reg = np.random.randn(100, 5)
y_reg = 3*X_reg[:, 0] + 2*X_reg[:, 1] + np.random.randn(100) * 5

# 7.2 Fit regular LinearRegression
# TODO: Fit

# 7.3 Fit Ridge regression (alpha=1.0)
# TODO: Fit Ridge

# 7.4 Compare coefficients
# TODO: Compare

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("\n--- EXERCISE 1: Simple Linear Regression ---")
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
scores = np.array([35, 42, 50, 55, 62, 68, 72, 78, 82, 88])

model = LinearRegression()
model.fit(hours.reshape(-1, 1), scores)

print(f"Model: score = {model.coef_[0]:.2f} * hours + {model.intercept_:.2f}")
print(f"Score for 11 hours: {model.predict([[11]])[0]:.2f}")
print(f"R² Score: {r2_score(scores, model.predict(hours.reshape(-1, 1))):.4f}")
rmse = np.sqrt(mean_squared_error(scores, model.predict(hours.reshape(-1, 1))))
print(f"RMSE: {rmse:.4f}")

print("\n--- EXERCISE 2: Multiple Linear Regression ---")
data = {
    'study_hours': [5, 3, 7, 2, 8, 4, 6, 1, 9, 5],
    'sleep_hours': [7, 6, 6, 5, 7, 6, 7, 4, 8, 6],
    'attendance': [90, 80, 95, 70, 100, 85, 92, 65, 98, 88],
    'score': [75, 60, 85, 50, 92, 68, 80, 45, 95, 72]
}
df = pd.DataFrame(data)
X = df[['study_hours', 'sleep_hours', 'attendance']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")
print(f"R² Score: {r2_score(y_test, model.predict(X_test)):.4f}")

print("\n--- EXERCISE 3: Gradient Descent ---")
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = 0
    b = 0
    n = len(X)

    for _ in range(iterations):
        y_pred = m * X + b
        error = y - y_pred
        m = m + learning_rate * (2/n) * np.sum(error * X)
        b = b + learning_rate * (2/n) * np.sum(error)

    return m, b

X_test = np.array([1, 2, 3, 4, 5])
y_test = np.array([2, 4, 5, 4, 5])
X_norm = (X_test - np.mean(X_test)) / np.std(X_test)
m_gd, b_gd = gradient_descent(X_norm, y_test)
print(f"Gradient descent: m={m_gd:.4f}, b={b_gd:.4f}")

model_sklearn = LinearRegression()
model_sklearn.fit(X_test.reshape(-1, 1), y_test)
print(f"sklearn: m={model_sklearn.coef_[0]:.4f}, b={model_sklearn.intercept_:.4f}")

print("\n--- EXERCISE 4: Feature Scaling ---")
np.random.seed(42)
X1 = np.random.rand(100) * 1000
X2 = np.random.rand(100) * 10
X3 = np.random.rand(100)
y = 0.5 * X1 + 10 * X2 + 100 * X3 + np.random.randn(100) * 10

X_no_scale = np.column_stack([X1, X2, X3])
model_no_scale = LinearRegression()
model_no_scale.fit(X_no_scale, y)
print(f"Without scaling: {model_no_scale.coef_}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_scale)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)
print(f"With scaling: {model_scaled.coef_}")

print("\n--- EXERCISE 5: Polynomial Regression ---")
np.random.seed(42)
X_poly = np.linspace(0, 10, 50)
y_poly = X_poly**2 + np.random.randn(50) * 10

model_linear = LinearRegression()
model_linear.fit(X_poly.reshape(-1, 1), y_poly)
r2_linear = r2_score(y_poly, model_linear.predict(X_poly.reshape(-1, 1)))
print(f"Linear R²: {r2_linear:.4f}")

X_poly_features = np.column_stack([X_poly, X_poly**2])
model_poly = LinearRegression()
model_poly.fit(X_poly_features, y_poly)
r2_poly = r2_score(y_poly, model_poly.predict(X_poly_features))
print(f"Polynomial R²: {r2_poly:.4f}")

print("\n--- EXERCISE 6: House Price Prediction ---")
houses = pd.DataFrame({
    'size_sqft': [1400, 1600, 1700, 1875, 1100, 1550, 2100, 1750, 1600, 1450],
    'bedrooms': [3, 3, 2, 4, 2, 3, 4, 3, 3, 3],
    'age_years': [10, 15, 5, 8, 20, 12, 3, 7, 10, 15],
    'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
})

X = houses[['size_sqft', 'bedrooms', 'age_years']]
y = houses['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

new_house = [[2000, 4, 5]]
print(f"Prediction for new house: ${model.predict(new_house)[0]:,.0f}")
print(f"R² Score: {r2_score(y_test, model.predict(X_test)):.4f}")
print(f"Coefficients: {dict(zip(X.columns, model.coef_))}")

print("\n--- EXERCISE 7: Regularization ---")
np.random.seed(42)
X_reg = np.random.randn(100, 5)
y_reg = 3*X_reg[:, 0] + 2*X_reg[:, 1] + np.random.randn(100) * 5

model_lr = LinearRegression()
model_lr.fit(X_reg, y_reg)

model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_reg, y_reg)

print(f"Linear Regression coefficients: {model_lr.coef_}")
print(f"Ridge coefficients: {model_ridge.coef_}")
