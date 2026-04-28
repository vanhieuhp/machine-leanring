"""
Random Forest - Deep Dive
==========================

This module covers:
- Understanding Random Forest algorithm
- Building Random Forest from scratch
- Using scikit-learn RandomForestClassifier/Regressor
- Hyperparameter tuning
- Feature importance analysis
- OOB (Out-of-Bag) score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression

# ============================================================================
# 1. UNDERSTANDING RANDOM FOREST
# ============================================================================

print("=" * 70)
print("1. RANDOM FOREST CONCEPT")
print("=" * 70)

print("""
Random Forest is an ensemble method based on bagging:
- Bagging: Bootstrap Aggregating
- Train multiple decision trees on different bootstrap samples
- Each tree makes a prediction
- Final prediction = majority vote (classification) or average (regression)

Key Randomness in Random Forest:
1. Bootstrap Sampling: Each tree trains on ~63% of data (with replacement)
2. Feature Subset: At each split, only random subset of features is considered
""")

# ============================================================================
# 2. BUILDING RANDOM FOREST FROM SCRATCH (SIMPLIFIED)
# ============================================================================

print("\n" + "=" * 70)
print("2. RANDOM FOREST FROM SCRATCH")
print("=" * 70)

class SimpleDecisionTree:
    """Simple decision tree for demonstration"""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Base case: max depth reached or pure node
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        # Find best split
        best_split = {}
        best_gain = -np.inf

        n_samples, n_features = X.shape

        # Random feature subset
        features = np.random.choice(n_features, max(1, n_features // 2), replace=False)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate information gain
                gain = self._information_gain(y, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        # If no good split found, return mean
        if best_gain == -np.inf:
            return np.mean(y)

        # Recursively build children
        left_tree = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _information_gain(self, y, left_mask, right_mask):
        def gini_impurity(labels):
            if len(labels) == 0:
                return 0
            probs = np.bincount(labels) / len(labels)
            return 1 - np.sum(probs ** 2)

        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            return 0

        parent_gini = gini_impurity(y)
        child_gini = (n_left / n) * gini_impurity(y[left_mask]) + \
                     (n_right / n) * gini_impurity(y[right_mask])

        return parent_gini - child_gini

    def predict_single(self, x):
        node = self.tree
        while isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


class SimpleRandomForest:
    """Simple Random Forest implementation"""
    def __init__(self, n_estimators=10, max_depth=5, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Train tree
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        # Collect all tree predictions
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Average (for regression) or majority vote (for classification)
        return np.mean(predictions, axis=0)


# Test with synthetic data
print("\nTesting Simple Random Forest:")
X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_simple = SimpleRandomForest(n_estimators=5, max_depth=3, random_state=42)
rf_simple.fit(X_train, y_train)
y_pred_simple = rf_simple.predict(X_test)

print(f"Simple RF R² Score: {r2_score(y_test, y_pred_simple):.4f}")

# ============================================================================
# 3. USING SCIKIT-LEARN RANDOM FOREST
# ============================================================================

print("\n" + "=" * 70)
print("3. USING SCIKIT-LEARN")
print("=" * 70)

# Generate classification data
X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

print("\nRandom Forest Classifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================================
# 4. HYPERPARAMETERS EXPLAINED
# ============================================================================

print("\n" + "=" * 70)
print("4. KEY HYPERPARAMETERS")
print("=" * 70)

print("""
n_estimators: Number of trees in the forest
- More trees = better performance but slower
- Typical: 100-1000
- Start with 100, increase if needed

max_depth: Maximum depth of each tree
- Deeper = more complex = potential overfitting
- Typical: 10-30 or None (unlimited)
- Start with None, reduce if overfitting

min_samples_split: Minimum samples to split a node
- Higher = less complex trees
- Typical: 2-10

min_samples_leaf: Minimum samples in leaf node
- Higher = less complex trees
- Typical: 1-10

max_features: Features to consider for best split
- 'sqrt': sqrt(n_features)
- 'log2': log2(n_features)
- Typical: 'sqrt' for classification
""")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("5. FEATURE IMPORTANCE")
print("=" * 70)

# Get feature importance
feature_importance = rf_classifier.feature_importances_
feature_names = [f'Feature {i}' for i in range(X.shape[1])]

print("\nFeature Importance:")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.4f}")

# Visualize
plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importance, color='forestgreen')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# ============================================================================
# 6. OOB (OUT-OF-BAG) SCORE
# ============================================================================

print("\n" + "=" * 70)
print("6. OOB (OUT-OF-BAG) SCORE")
print("=" * 70)

print("""
OOB Score:
- Each tree trains on ~63% of data (bootstrap)
- The remaining ~37% is "OOB" - can be used for validation
- No separate validation set needed!
""")

# Train with OOB score
rf_with_oob = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    oob_score=True,
    random_state=42
)
rf_with_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_with_oob.oob_score_:.4f}")
print(f"Test Accuracy: {rf_with_oob.score(X_test, y_test):.4f}")

# ============================================================================
# 7. RANDOM FOREST FOR REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("7. RANDOM FOREST FOR REGRESSION")
print("=" * 70)

# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=10,
    noise=10,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_regressor.predict(X_test_reg)

print("\nRandom Forest Regressor Results:")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")

# Feature importance for regression
feature_importance_reg = rf_regressor.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh([f'Feature {i}' for i in range(10)], feature_importance_reg, color='steelblue')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance (Regression)')
plt.tight_layout()
plt.show()

# ============================================================================
# 8. PRACTICAL EXAMPLE: IRIS DATASET
# ============================================================================

print("\n" + "=" * 70)
print("8. PRACTICAL EXAMPLE: IRIS DATASET")
print("=" * 70)

from sklearn.datasets import load_iris

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Train Random Forest
rf_iris = RandomForestClassifier(n_estimators=100, random_state=42)
rf_iris.fit(X_iris, y_iris)

# Cross-validation
cv_scores = cross_val_score(rf_iris, X_iris, y_iris, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
print("\nIris Feature Importance:")
for name, importance in zip(iris.feature_names, rf_iris.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# ============================================================================
# 9. COMPARING DIFFERENT CONFIGURATIONS
# ============================================================================

print("\n" + "=" * 70)
print("9. COMPARING CONFIGURATIONS")
print("=" * 70)

configs = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 100, 'max_depth': None},
    {'n_estimators': 200, 'max_depth': 15},
]

print("\nComparing different Random Forest configurations:")
print("-" * 50)

for config in configs:
    rf = RandomForestClassifier(**config, random_state=42)
    rf.fit(X_train, y_train)
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"n_est={config['n_estimators']:3d}, max_depth={str(config['max_depth']):4s} | "
          f"Train: {train_score:.4f}, Test: {test_score:.4f}")

# ============================================================================
# 10. ADVANCED: EXTRA TREES
# ============================================================================

print("\n" + "=" * 70)
print("10. EXTRA TREES (EXTREMELY RANDOMIZED TREES)")
print("=" * 70)

print("""
Extra Trees vs Random Forest:
- Random Forest: finds best split among random subset
- Extra Trees: uses random splits (even more random)
- Faster training (no searching for best split)
- More diverse trees = potentially better generalization
""")

from sklearn.ensemble import ExtraTreesClassifier

et_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_classifier.fit(X_train, y_train)

print(f"Extra Trees Accuracy: {et_classifier.score(X_test, y_test):.4f}")
print(f"Random Forest Accuracy: {rf_classifier.score(X_test, y_test):.4f}")

print("\n" + "=" * 70)
print("RANDOM FOREST SUMMARY")
print("=" * 70)
print("""
Key Takeaways:
1. Random Forest = Bagging + Decision Trees + Feature Randomness
2. Use n_estimators=100-500, max_depth=10-30 as starting point
3. OOB score provides built-in validation
4. Feature importance is built-in and useful
5. Less prone to overfitting than single decision tree
6. Works for both classification and regression

When to use:
- As baseline model
- When you need feature importance
- When data has missing values
- For medium-sized datasets
""")
