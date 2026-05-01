"""
SVM for Regression
==================

This module covers:
- SVM Regression (SVR)
- Epsilon-insensitive loss
- Using SVR with different kernels
- Hyperparameter tuning
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression, make_blobs
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. SVM REGRESSION CONCEPT
# ============================================================================

print("=" * 70)
print("1. SVM REGRESSION CONCEPT")
print("=" * 70)

print("""
SVM for Regression (SVR):
- Instead of separating classes, SVR fits a tube around the data
- Points inside the tube incur no loss
- Only points outside the tube contribute to the loss

Key Parameters:
1. epsilon (ε): Width of the tube
   - Points within ε distance from prediction have zero loss
   - Larger ε = wider tube = more points inside = simpler model
   - Smaller ε = narrower tube = more points outside = complex model

2. C: Regularization parameter
   - Controls trade-off between flatness of curve and tolerance for deviations
   - Similar to classification: high C = more complex, low C = simpler

The Goal:
- Find a function with maximum flatness (small weights)
- While allowing deviations up to ε
""")

# ============================================================================
# 2. SIMPLE SVR EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("2. SIMPLE SVR EXAMPLE")
print("=" * 70)

# Generate simple 1D data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale target for better performance
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Train SVR with different kernels
kernels = ['linear', 'rbf', 'poly']

plt.figure(figsize=(15, 5))

for i, kernel in enumerate(kernels):
    svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_train)

    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
    plt.plot(X_train, y_pred, color='red', linewidth=2, label=f'SVR ({kernel})')
    plt.title(f'Kernel: {kernel}')
    plt.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# 3. EPSILON PARAMETER
# ============================================================================

print("\n" + "=" * 70)
print("3. EPSILON PARAMETER")
print("=" * 70)

print("""
Epsilon (ε) controls the insensitivity zone:
- Points within ε of the prediction have zero loss
- Creates a "tube" around the regression line

Effect of epsilon:
- ε = 0.1: Narrow tube, more points outside, complex model
- ε = 0.5: Wider tube, more points inside, simpler model
- ε = 1.0: Very wide tube, almost just mean prediction
""")

# Compare epsilon values
epsilons = [0.01, 0.1, 0.5, 1.0]

plt.figure(figsize=(15, 5))

for i, epsilon in enumerate(epsilons):
    svr = SVR(kernel='rbf', C=1.0, epsilon=epsilon)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_train)

    plt.subplot(1, 4, i + 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.5)
    plt.plot(X_train, y_pred, color='red', linewidth=2)
    plt.title(f'ε = {epsilon}')
    plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# ============================================================================
# 4. C PARAMETER FOR REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("4. C PARAMETER FOR REGRESSION")
print("=" * 70)

print("""
C in SVR:
- Controls regularization trade-off
- Low C: More emphasis on flatness (simpler model)
- High C: More emphasis on fitting data (complex model)

This is opposite to the intuition in some ways:
- High C = small margin = more complex
- Low C = large margin = simpler
""")

# Compare C values
C_values = [0.1, 1, 10, 100]

plt.figure(figsize=(15, 5))

for i, C in enumerate(C_values):
    svr = SVR(kernel='rbf', C=C, epsilon=0.1)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_train)

    plt.subplot(1, 4, i + 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.5)
    plt.plot(X_train, y_pred, color='red', linewidth=2)
    plt.title(f'C = {C}')
    plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# ============================================================================
# 5. PRACTICAL EXAMPLE: MULTI-FEATURE REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("5. MULTI-FEATURE REGRESSION")
print("=" * 70)

# Generate regression data
X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVR
svr = SVR(kernel='rbf', C=10, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)

print(f"\nSVR Results:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# ============================================================================
# 6. SVR WITH DIFFERENT KERNELS
# ============================================================================

print("\n" + "=" * 70)
print("6. KERNELS FOR REGRESSION")
print("=" * 70)

print("""
Kernels for SVR:
1. Linear: K(x, y) = x · y
   - Works like linear regression but with epsilon tube

2. RBF: K(x, y) = exp(-γ||x - y||²)
   - Most commonly used
   - Creates non-linear decision boundary
   - γ (gamma): controls flexibility

3. Polynomial: K(x, y) = (x · y + c)^d
   - Creates polynomial relationships
   - d: degree
""")

# Compare kernels
kernels_results = {}

for kernel in ['linear', 'rbf', 'poly']:
    svr = SVR(kernel=kernel, C=10, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    kernels_results[kernel] = {'r2': r2, 'rmse': rmse}

    print(f"\n{kernel.capitalize()} Kernel:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")

# ============================================================================
# 7. GAMMA PARAMETER
# ============================================================================

print("\n" + "=" * 70)
print("7. GAMMA PARAMETER (RBF)")
print("=" * 70)

print("""
Gamma (γ) in RBF Kernel:
- Defines how far the influence of a single training example reaches
- Low γ: far reach - each point influences many others
- High γ: short reach - each point only influences nearby ones

Analogy:
- Low γ: Smooth function, large influence radius
- High γ: Complex function, small influence radius
""")

# Compare gamma values
gammas = [0.01, 0.1, 1, 10]

plt.figure(figsize=(15, 5))

for i, gamma in enumerate(gammas):
    svr = SVR(kernel='rbf', C=10, epsilon=0.1, gamma=gamma)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_train)

    plt.subplot(1, 4, i + 1)
    plt.scatter(X_train[:, 0], y_train, color='blue', alpha=0.5)
    plt.scatter(X_train[:, 0], y_pred, color='red', s=10)
    plt.title(f'γ = {gamma}')
    plt.xlabel('Feature 1')
    plt.ylabel('Target')

plt.tight_layout()
plt.show()

# ============================================================================
# 8. COMPARISON: SVR vs LINEAR REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("8. COMPARISON: SVR vs LINEAR REGRESSION")
print("=" * 70)

from sklearn.linear_model import LinearRegression

# Simple linear data
X_linear = np.linspace(0, 10, 100).reshape(-1, 1)
y_linear = 3 * X_linear.flatten() + 5 + np.random.normal(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

# Linear Regression
()
lr.fit(Xlr = LinearRegression_train, y_train)
y_pred_lr = lr.predict(X_test)

# SVR
svr = SVR(kernel='linear', C=10, epsilon=0.1)
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

print("\nLinear Regression:")
print(f"  R² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")

print("\nSVR (Linear):")
print(f"  R² Score: {r2_score(y_test, y_pred_svr):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.4f}")

# Visualize
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_lr, color='green', linewidth=2, label='Linear Regression')
plt.plot(X_test, y_pred_svr, color='red', linewidth=2, label='SVR')
plt.legend()
plt.title('Linear Regression vs SVR')
plt.show()

# ============================================================================
# 9. HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "=" * 70)
print("9. HYPERPARAMETER TUNING")
print("=" * 70)

from sklearn.model_selection import GridSearchCV

print("""
Common parameters to tune:
1. C: Regularization (0.1 to 100)
2. epsilon: Tube width (0.01 to 1)
3. gamma: RBF parameter (scale, auto, or specific value)
""")

# Grid search (small for demo)
param_grid = {
    'C': [1, 10],
    'epsilon': [0.1, 0.5],
    'gamma': ['scale', 'auto']
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Evaluate on test
y_pred_best = grid_search.best_estimator_.predict(X_test_scaled)
print(f"Test R² Score: {r2_score(y_test, y_pred_best):.4f}")

print("\n" + "=" * 70)
print("SVM REGRESSION SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. SVR fits a tube around the data
2. epsilon controls tube width
3. C controls regularization
4. RBF kernel is most common
5. Always scale features!

When to use SVR:
- When you need robust regression
- When data has non-linear patterns
- When you want to control model complexity

When NOT to use SVR:
- Very large datasets (slow)
- Many features (slow)
- When interpretability is critical
""")
