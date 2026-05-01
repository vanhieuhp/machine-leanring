"""
Gradient Boosting - Deep Dive
==============================

This module covers:
- Understanding Gradient Boosting algorithm
- Gradient Boosting from sklearn
- Key hyperparameters
- Comparison with Random Forest
- Preventing overfitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.datasets import make_classification, make_regression

# ============================================================================
# 1. UNDERSTANDING GRADIENT BOOSTING
# ============================================================================

print("=" * 70)
print("1. GRADIENT BOOSTING CONCEPT")
print("=" * 70)

print("""
Gradient Boosting: Sequential ensemble that learns from mistakes

Key Ideas:
1. Start with simple model (usually predicting mean)
2. Calculate residuals (errors) from current predictions
3. Fit new model to predict these residuals
4. Add new model to ensemble (with learning rate)
5. Repeat for n_estimators

Analogy:
- First model: "Predict house price = $200k" (very rough)
- Second model learns: "The house has 3 bedrooms, add $50k"
- Third model learns: "House is near school, add $20k"
- Final: Sum of all corrections
""")

# ============================================================================
# 2. GRADIENT BOOSTING REGRESSION FROM SCRATCH
# ============================================================================

print("\n" + "=" * 70)
print("2. GRADIENT BOOSTING FROM SCRATCH")
print("=" * 70)

class SimpleGradientBoosting:
    """Simplified Gradient Boosting for demonstration"""
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Initial prediction (mean for regression)
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(len(y), self.initial_prediction)

        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient)
            residuals = y - current_prediction

            # Fit a decision tree to residuals
            tree = SimpleDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Make predictions
            tree_predictions = tree.predict(X)

            # Update current prediction
            current_prediction += self.learning_rate * tree_predictions

            # Store the tree
            self.models.append(tree)

        return self

    def predict(self, X):
        # Start with initial prediction
        predictions = np.full(len(X), self.initial_prediction)

        # Add contributions from each tree
        for tree in self.models:
            predictions += self.learning_rate * tree.predict(X)

        return predictions


class SimpleDecisionTreeRegressor:
    """Simple decision tree for regression"""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 5:
            return np.mean(y)

        best_split = {}
        best_gain = -np.inf

        n_samples, n_features = X.shape

        for feature in np.random.choice(n_features, max(1, n_features // 2), replace=False):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue

                # MSE reduction
                parent_mse = np.mean((y - np.mean(y)) ** 2)
                left_mse = np.mean((y[left_mask] - np.mean(y[left_mask])) ** 2)
                right_mse = np.mean((y[right_mask] - np.mean(y[right_mask])) ** 2)

                gain = parent_mse - (np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse) / len(y)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        if not best_split:
            return np.mean(y)

        left_tree = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])


# Test custom implementation
X, y = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_simple = SimpleGradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3)
gb_simple.fit(X_train, y_train)
y_pred_simple = gb_simple.predict(X_test)

print(f"\nSimple GB R² Score: {r2_score(y_test, y_pred_simple):.4f}")

# ============================================================================
# 3. USING SCIKIT-LEARN GRADIENT BOOSTING
# ============================================================================

print("\n" + "=" * 70)
print("3. USING SCIKIT-LEARN")
print("=" * 70)

# Classification
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

gb_classifier.fit(X_train, y_train)
y_pred = gb_classifier.predict(X_test)

print("\nGradient Boosting Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================================
# 4. KEY HYPERPARAMETERS
# ============================================================================

print("\n" + "=" * 70)
print("4. KEY HYPERPARAMETERS")
print("=" * 70)

print("""
n_estimators: Number of boosting stages
- More trees = more complex model
- Use with learning_rate (shrinkage)
- Typical: 100-500

learning_rate (shrinkage): How much each tree contributes
- Lower = slower learning but more trees = better generalization
- Typical: 0.01 - 0.3
- Common setting: 0.1

max_depth: Maximum depth of each tree
- Keep small (3-10) to prevent overfitting
- Default: 3

min_samples_split: Minimum samples to split
- Prevents overfitting
- Typical: 5-20

min_samples_leaf: Minimum samples in leaf
- Typical: 1-10

subsample: Fraction of samples for each tree
- < 1.0 = stochastic gradient boosting
- Typical: 0.8
- Helps reduce overfitting
""")

# ============================================================================
# 5. LEARNING RATE VS N_ESTIMATORS
# ============================================================================

print("\n" + "=" * 70)
print("5. LEARNING RATE VS N_ESTIMATORS")
print("=" * 70)

plt.figure(figsize=(12, 6))

train_sizes = [10, 25, 50, 75, 100, 150, 200]
learning_rates = [0.01, 0.05, 0.1, 0.2]

for lr in learning_rates:
    train_scores = []
    test_scores = []

    for n_est in train_sizes:
        gb = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=3,
            random_state=42
        )
        gb.fit(X_train, y_train)
        train_scores.append(gb.score(X_train, y_train))
        test_scores.append(gb.score(X_test, y_test))

    plt.plot(train_sizes, test_scores, marker='o', label=f'lr={lr}')

plt.xlabel('Number of Estimators')
plt.ylabel('Test Accuracy')
plt.title('Learning Rate vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("""
Key Insight:
- Lower learning rate requires more estimators
- Trade-off: more trees = slower training but better generalization
- Use early_stopping to find optimal n_estimators
""")

# ============================================================================
# 6. EARLY STOPPING
# ============================================================================

print("\n" + "=" * 70)
print("6. EARLY STOPPING")
print("=" * 70)

print("""
Early Stopping:
- Monitor validation score during training
- Stop when no improvement for n rounds
- Prevents overfitting
- Finds optimal number of estimators
""")

gb_early = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.1,
    n_iter_no_change=10,  # Stop if no improvement for 10 rounds
    random_state=42
)

gb_early.fit(X_train, y_train)

print(f"Optimal n_estimators: {gb_early.n_estimators_}")
print(f"Test Accuracy: {gb_early.score(X_test, y_test):.4f}")

# ============================================================================
# 7. STAGING PREDICTION (UNDERSTANDING BOOSTING)
# ============================================================================

print("\n" + "=" * 70)
print("7. STAGING PREDICTION")
print("=" * 70)

# See how model improves with each stage
gb_staged = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_staged.fit(X_train, y_train)

train_scores = []
test_scores = []

for i in range(1, 101):
    train_scores.append(gb_staged.score(X_train[:i*5] if i*5 <= len(X_train) else X_train,
                                         y_train[:i*5] if i*5 <= len(y_train) else y_train))
    test_scores.append(gb_staged.staged_predict(X_test))

# Plot learning curve
plt.figure(figsize=(10, 5))
staged_test_scores = [gb_staged.score(X_test, y_test) for gb_staged in
                      [GradientBoostingClassifier(n_estimators=i, random_state=42).fit(X_train, y_train)
                       for i in range(1, 101)]]

plt.plot(range(1, 101), staged_test_scores, color='green')
plt.xlabel('Number of Estimators')
plt.ylabel('Test Accuracy')
plt.title('Gradient Boosting: Test Accuracy vs n_estimators')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("8. FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = gb_classifier.feature_importances_

plt.figure(figsize=(10, 5))
plt.barh([f'Feature {i}' for i in range(10)], feature_importance, color='orange')
plt.xlabel('Importance')
plt.title('Gradient Boosting Feature Importance')
plt.tight_layout()
plt.show()

# ============================================================================
# 9. GRADIENT BOOSTING FOR REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("9. GRADIENT BOOSTING FOR REGRESSION")
print("=" * 70)

X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

gb_regressor = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = gb_regressor.predict(X_test_reg)

print(f"\nGradient Boosting Regressor:")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")

# ============================================================================
# 10. COMPARISON: RANDOM FOREST VS GRADIENT BOOSTING
# ============================================================================

print("\n" + "=" * 70)
print("10. RANDOM FOREST vs GRADIENT BOOSTING")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier

# Train both models
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

print("\nComparison on Classification:")
print("-" * 50)
print(f"Random Forest:       {rf_model.score(X_test, y_test):.4f}")
print(f"Gradient Boosting:    {gb_model.score(X_test, y_test):.4f}")

# Training time comparison
import time

start = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start

start = time.time()
gb_model.fit(X_train, y_train)
gb_time = time.time() - start

print(f"\nTraining Time:")
print("-" * 50)
print(f"Random Forest:       {rf_time:.3f}s")
print(f"Gradient Boosting:  {gb_time:.3f}s")

print("""
Summary of Differences:

| Aspect          | Random Forest | Gradient Boosting |
|-----------------|---------------|------------------|
| Training        | Parallel      | Sequential       |
| Speed           | Faster        | Slower           |
| Overfitting     | Less likely   | More prone       |
| Hyperparameters | More robust   | Need tuning      |
| Accuracy        | Good          | Often better     |

When to use:
- Random Forest: Quick baseline, less tuning
- Gradient Boosting: Maximum accuracy, well-tuned
""")

print("\n" + "=" * 70)
print("GRADIENT BOOSTING SUMMARY")
print("=" * 70)
print("""
Key Takeaways:
1. Sequential ensemble that learns from errors
2. Start with learning_rate=0.1, tune n_estimators
3. Use early_stopping to prevent overfitting
4. Smaller max_depth (3-5) works well
5. subsample < 1 adds regularization
6. Generally more accurate but slower than Random Forest

Next Steps:
- Learn XGBoost/LightGBM for optimized implementations
- Understand stacking for combining models
""")
