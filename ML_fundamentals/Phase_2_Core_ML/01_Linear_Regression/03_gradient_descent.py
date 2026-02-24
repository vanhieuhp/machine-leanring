"""
Linear Regression - Part 3: Gradient Descent
==============================================

This module covers:
- Gradient descent algorithm
- Learning rate
- Convergence
- Batch vs Stochastic gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ============================================================================
# 1. GRADIENT DESCENT FROM SCRATCH
# ============================================================================

print("=" * 70)
print("1. GRADIENT DESCENT FROM SCRATCH")
print("=" * 70)

# Generate data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 2 * X + 1 + np.random.randn(10) * 2

# Normalize features
X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

# Initialize parameters
m = 0.0  # slope
b = 0.0  # intercept
learning_rate = 0.01
iterations = 1000
n = len(X)

# Store cost history
cost_history = []

print(f"Initial: m={m:.4f}, b={b:.4f}")

# Gradient descent
for i in range(iterations):
    # Predictions
    y_pred = m * X_normalized + b

    # Calculate gradients
    dm = (-2/n) * np.sum(X_normalized * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

    # Calculate cost (MSE)
    cost = np.mean((y - y_pred) ** 2)
    cost_history.append(cost)

    if (i + 1) % 100 == 0:
        print(f"Iteration {i+1}: m={m:.4f}, b={b:.4f}, Cost={cost:.4f}")

print(f"\nFinal: m={m:.4f}, b={b:.4f}")

# ============================================================================
# 2. LEARNING RATE EFFECT
# ============================================================================

print("\n" + "=" * 70)
print("2. LEARNING RATE EFFECT")
print("=" * 70)

learning_rates = [0.001, 0.01, 0.1, 0.5]
results = {}

for lr in learning_rates:
    m = 0.0
    b = 0.0
    cost_hist = []

    for i in range(1000):
        y_pred = m * X_normalized + b
        dm = (-2/n) * np.sum(X_normalized * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        m = m - lr * dm
        b = b - lr * db
        cost = np.mean((y - y_pred) ** 2)
        cost_hist.append(cost)

    results[lr] = cost_hist
    print(f"Learning rate {lr}: Final cost = {cost_hist[-1]:.4f}")

# ============================================================================
# 3. CONVERGENCE VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("3. CONVERGENCE VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Cost vs Iterations
for lr, cost_hist in results.items():
    axes[0].plot(cost_hist, label=f'LR={lr}', linewidth=2)

axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Cost (MSE)')
axes[0].set_title('Convergence with Different Learning Rates')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Plot 2: Learning rate comparison (zoomed)
for lr, cost_hist in results.items():
    axes[1].plot(cost_hist[:100], label=f'LR={lr}', linewidth=2)

axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Cost (MSE)')
axes[1].set_title('First 100 Iterations')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 4. BATCH VS STOCHASTIC GRADIENT DESCENT
# ============================================================================

print("\n" + "=" * 70)
print("4. BATCH VS STOCHASTIC GRADIENT DESCENT")
print("=" * 70)

# Batch Gradient Descent (BGD)
def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = 0.0
    b = 0.0
    n = len(X)
    cost_history = []

    for i in range(iterations):
        y_pred = m * X + b
        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        m = m - learning_rate * dm
        b = b - learning_rate * db
        cost = np.mean((y - y_pred) ** 2)
        cost_history.append(cost)

    return m, b, cost_history

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = 0.0
    b = 0.0
    n = len(X)
    cost_history = []

    for i in range(iterations):
        # Random sample
        idx = np.random.randint(0, n)
        x_i = X[idx]
        y_i = y[idx]

        y_pred = m * x_i + b
        dm = -2 * x_i * (y_i - y_pred)
        db = -2 * (y_i - y_pred)
        m = m - learning_rate * dm
        b = b - learning_rate * db

        # Calculate full cost
        y_pred_full = m * X + b
        cost = np.mean((y - y_pred_full) ** 2)
        cost_history.append(cost)

    return m, b, cost_history

# Mini-batch Gradient Descent
def mini_batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000, batch_size=4):
    m = 0.0
    b = 0.0
    n = len(X)
    cost_history = []

    for i in range(iterations):
        # Random batch
        indices = np.random.choice(n, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]

        y_pred = m * X_batch + b
        dm = (-2/batch_size) * np.sum(X_batch * (y_batch - y_pred))
        db = (-2/batch_size) * np.sum(y_batch - y_pred)
        m = m - learning_rate * dm
        b = b - learning_rate * db

        # Calculate full cost
        y_pred_full = m * X + b
        cost = np.mean((y - y_pred_full) ** 2)
        cost_history.append(cost)

    return m, b, cost_history

# Compare methods
m_bgd, b_bgd, cost_bgd = batch_gradient_descent(X_normalized, y, learning_rate=0.01, iterations=500)
m_sgd, b_sgd, cost_sgd = stochastic_gradient_descent(X_normalized, y, learning_rate=0.01, iterations=500)
m_mb, b_mb, cost_mb = mini_batch_gradient_descent(X_normalized, y, learning_rate=0.01, iterations=500, batch_size=4)

print(f"BGD: m={m_bgd:.4f}, b={b_bgd:.4f}, Final cost={cost_bgd[-1]:.4f}")
print(f"SGD: m={m_sgd:.4f}, b={b_sgd:.4f}, Final cost={cost_sgd[-1]:.4f}")
print(f"Mini-batch: m={m_mb:.4f}, b={b_mb:.4f}, Final cost={cost_mb[-1]:.4f}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cost_bgd, label='Batch GD', linewidth=2)
plt.plot(cost_sgd, label='Stochastic GD', linewidth=2, alpha=0.7)
plt.plot(cost_mb, label='Mini-batch GD', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Methods Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(cost_bgd[:100], label='Batch GD', linewidth=2)
plt.plot(cost_sgd[:100], label='Stochastic GD', linewidth=2, alpha=0.7)
plt.plot(cost_mb[:100], label='Mini-batch GD', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('First 100 Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 5. PRACTICAL INSIGHTS
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL INSIGHTS")
print("=" * 70)

print("""
Batch Gradient Descent (BGD):
  - Uses all data for each update
  - Smooth convergence
  - Slow for large datasets
  - Guaranteed to converge (convex)

Stochastic Gradient Descent (SGD):
  - Uses one sample per update
  - Noisy convergence
  - Fast for large datasets
  - Can escape local minima

Mini-batch Gradient Descent:
  - Best of both worlds
  - Uses small batches
  - Good balance of speed and stability
  - Most commonly used in practice

Learning Rate:
  - Too small: Slow convergence
  - Too large: Divergence or oscillation
  - Adaptive methods: Adam, RMSprop adjust automatically
""")
