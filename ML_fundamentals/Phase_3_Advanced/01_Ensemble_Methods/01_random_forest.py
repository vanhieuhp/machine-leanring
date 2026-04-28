"""
=================================================================
01 - RANDOM FOREST: From Theory to Practice
=================================================================
Topics covered:
  1. Decision Tree refresher
  2. Bagging concept
  3. Random Forest from scratch (simplified)
  4. Random Forest with scikit-learn
  5. Feature importance analysis
  6. OOB (Out-of-Bag) evaluation
  7. Hyperparameter tuning
=================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification, load_wine
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")

# ── Section 1: Why Ensembles? Single Tree vs Forest ──────────────
print("=" * 65)
print("SECTION 1: Single Decision Tree vs Random Forest")
print("=" * 65)

# Create a moderately complex dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Single Decision Tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, single_tree.predict(X_test))
tree_cv = cross_val_score(single_tree, X_train, y_train, cv=5).mean()

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_cv = cross_val_score(rf, X_train, y_train, cv=5).mean()

print(f"\nSingle Decision Tree:")
print(f"  Test Accuracy:  {tree_acc:.4f}")
print(f"  CV Accuracy:    {tree_cv:.4f}")
print(f"\nRandom Forest (100 trees):")
print(f"  Test Accuracy:  {rf_acc:.4f}")
print(f"  CV Accuracy:    {rf_cv:.4f}")
print(f"\n💡 Improvement: {(rf_acc - tree_acc) * 100:.1f}% on test set")

# ── Section 2: Bagging Concept ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Understanding Bagging")
print("=" * 65)

# Demonstrate how bagging works
print("\n📦 Bootstrap Sampling Demonstration:")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

np.random.seed(42)
for i in range(3):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    oob_samples = np.setdiff1d(data, bootstrap_sample)
    print(f"\n  Bootstrap Sample {i + 1}: {bootstrap_sample}")
    print(f"  OOB samples:          {oob_samples}")
    print(
        f"  % data used: {len(np.unique(bootstrap_sample)) / len(data) * 100:.1f}%"
    )

# BaggingClassifier demonstration
print("\n\n📊 Bagging with Different Base Estimators:")
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42,
)
bagging.fit(X_train, y_train)
bag_acc = accuracy_score(y_test, bagging.predict(X_test))
print(f"  Bagging (50 shallow trees): {bag_acc:.4f}")

# ── Section 3: Random Forest In-Depth ────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Random Forest — In-Depth Analysis")
print("=" * 65)

# Effect of number of trees
print("\n🌲 Effect of Number of Trees:")
n_trees_list = [1, 5, 10, 25, 50, 100, 200, 500]
accuracies = []

for n_trees in n_trees_list:
    rf_temp = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf_temp.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf_temp.predict(X_test))
    accuracies.append(acc)
    print(f"  n_estimators={n_trees:>3d}: accuracy = {acc:.4f}")

print("\n  ✦ Notice: accuracy improves rapidly then plateaus")
print("  ✦ More trees = diminishing returns but never hurts (just slower)")

# ── Section 4: Feature Importance ─────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Feature Importance")
print("=" * 65)

# Use Wine dataset for interpretability
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42
)

rf_wine = RandomForestClassifier(n_estimators=200, random_state=42)
rf_wine.fit(X_w_train, y_w_train)

print(f"\nWine Classification Accuracy: {accuracy_score(y_w_test, rf_wine.predict(X_w_test)):.4f}")

# Method 1: Mean Decrease Impurity (MDI)
print("\n📊 Method 1: Mean Decrease Impurity (built-in):")
mdi_importance = rf_wine.feature_importances_
sorted_idx = np.argsort(mdi_importance)[::-1]

for i in range(min(5, len(wine.feature_names))):
    idx = sorted_idx[i]
    print(f"  {i + 1}. {wine.feature_names[idx]:>25s}: {mdi_importance[idx]:.4f}")

# Method 2: Permutation Importance (more reliable)
print("\n📊 Method 2: Permutation Importance (recommended):")
perm_importance = permutation_importance(
    rf_wine, X_w_test, y_w_test, n_repeats=10, random_state=42
)
sorted_idx_perm = perm_importance.importances_mean.argsort()[::-1]

for i in range(min(5, len(wine.feature_names))):
    idx = sorted_idx_perm[i]
    print(
        f"  {i + 1}. {wine.feature_names[idx]:>25s}: "
        f"{perm_importance.importances_mean[idx]:.4f} "
        f"± {perm_importance.importances_std[idx]:.4f}"
    )

# ── Section 5: OOB Score ──────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Out-of-Bag (OOB) Evaluation")
print("=" * 65)

rf_oob = RandomForestClassifier(
    n_estimators=200, oob_score=True, random_state=42
)
rf_oob.fit(X_w_train, y_w_train)

test_acc = accuracy_score(y_w_test, rf_oob.predict(X_w_test))
oob_acc = rf_oob.oob_score_

print(f"\n  OOB Score:     {oob_acc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Difference:    {abs(oob_acc - test_acc):.4f}")
print("\n  💡 OOB score is a free cross-validation estimate!")
print("     No need for a separate validation set.")

# ── Section 6: Hyperparameter Tuning ──────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Hyperparameter Tuning with GridSearchCV")
print("=" * 65)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}

print(f"\n  Search space: {2 * 3 * 2 * 2} combinations × 3-fold CV")
print("  Running GridSearchCV... (this may take a moment)")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_w_train, y_w_train)

print(f"\n  Best Parameters: {grid_search.best_params_}")
print(f"  Best CV Score:   {grid_search.best_score_:.4f}")
print(
    f"  Test Accuracy:   "
    f"{accuracy_score(y_w_test, grid_search.predict(X_w_test)):.4f}"
)

# ── Section 7: Random Forest for Regression ───────────────────────
print("\n" + "=" * 65)
print("SECTION 7: Random Forest for Regression")
print("=" * 65)

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

X_reg, y_reg = make_regression(
    n_samples=500, n_features=10, n_informative=5, noise=20, random_state=42
)
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_r_train, y_r_train)
y_pred_reg = rf_reg.predict(X_r_test)

print(f"\n  R² Score:             {r2_score(y_r_test, y_pred_reg):.4f}")
print(f"  RMSE:                 {np.sqrt(mean_squared_error(y_r_test, y_pred_reg)):.4f}")
print(f"  OOB Score (R²):       {RandomForestRegressor(n_estimators=200, oob_score=True, random_state=42).fit(X_r_train, y_r_train).oob_score_:.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(
    """
✅ Key Takeaways:
  1. Random Forest = Bagging + Random Feature Selection
  2. More trees → better (with diminishing returns)
  3. Use OOB score for free validation
  4. Permutation importance > MDI importance
  5. Tune max_depth, min_samples_split, max_features
  6. Works for both classification and regression
  7. Rarely overfits (safe default choice)

📚 Next: 02_gradient_boosting.py (AdaBoost & Gradient Boosting)
"""
)
