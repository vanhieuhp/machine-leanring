"""
=================================================================
02 - GRADIENT BOOSTING: AdaBoost & Gradient Boosting Deep Dive
=================================================================
Topics covered:
  1. AdaBoost — algorithm and implementation
  2. Gradient Boosting — step by step
  3. Gradient Boosting Classifier & Regressor
  4. Learning rate and n_estimators trade-off
  5. Comparison: Bagging vs Boosting
  6. Practical tips for Gradient Boosting
=================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_wine
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
import warnings

warnings.filterwarnings("ignore")

# ── Section 1: AdaBoost ──────────────────────────────────────────
print("=" * 65)
print("SECTION 1: AdaBoost (Adaptive Boosting)")
print("=" * 65)

# Create dataset
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

# --- 1.1 AdaBoost with stumps (default) ---
print("\n📌 1.1 AdaBoost with Decision Stumps (depth=1):")
ada_stump = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42,
)
ada_stump.fit(X_train, y_train)
print(f"  Train Accuracy: {accuracy_score(y_train, ada_stump.predict(X_train)):.4f}")
print(f"  Test Accuracy:  {accuracy_score(y_test, ada_stump.predict(X_test)):.4f}")

# --- 1.2 AdaBoost with deeper trees ---
print("\n📌 1.2 AdaBoost with deeper trees (depth=3):")
ada_deep = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)
ada_deep.fit(X_train, y_train)
print(f"  Train Accuracy: {accuracy_score(y_train, ada_deep.predict(X_train)):.4f}")
print(f"  Test Accuracy:  {accuracy_score(y_test, ada_deep.predict(X_test)):.4f}")

# --- 1.3 Effect of n_estimators on AdaBoost ---
print("\n📌 1.3 AdaBoost: Effect of n_estimators:")
for n_est in [10, 50, 100, 200, 500]:
    ada = AdaBoostClassifier(
        n_estimators=n_est, learning_rate=0.5, random_state=42
    )
    ada.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, ada.predict(X_train))
    test_acc = accuracy_score(y_test, ada.predict(X_test))
    print(f"  n_estimators={n_est:>3d}: train={train_acc:.4f}, test={test_acc:.4f}")

# --- 1.4 Staged predictions (see how accuracy builds up) ---
print("\n📌 1.4 AdaBoost Staged Performance (accuracy after each round):")
ada_staged = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_staged.fit(X_train, y_train)

staged_scores = []
for i, y_pred in enumerate(ada_staged.staged_predict(X_test)):
    staged_scores.append(accuracy_score(y_test, y_pred))

checkpoints = [0, 9, 24, 49, 74, 99]
for idx in checkpoints:
    if idx < len(staged_scores):
        print(f"  After {idx + 1:>3d} rounds: {staged_scores[idx]:.4f}")

# ── Section 2: Gradient Boosting ──────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Gradient Boosting (Step-by-Step)")
print("=" * 65)

# --- 2.1 Manual demonstration of gradient boosting ---
print("\n📌 2.1 Manual Gradient Boosting (Regression Example):")
print("  Shows how residuals shrink at each step\n")

np.random.seed(42)
X_demo = np.random.rand(6, 1) * 10
y_demo = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])

# Step 0: initial prediction = mean
pred = np.full_like(y_demo, y_demo.mean())
print(f"  y_true:    {y_demo}")
print(f"  Step 0 (mean): {pred}")
print(f"  Residuals: {y_demo - pred}")
print(f"  MSE:       {np.mean((y_demo - pred) ** 2):.2f}")

learning_rate = 0.3
for step in range(1, 4):
    residuals = y_demo - pred
    # Simplified: pretend our "tree" learns the residuals perfectly
    tree_pred = residuals * 0.9  # a little imperfect
    pred = pred + learning_rate * tree_pred
    mse = np.mean((y_demo - pred) ** 2)
    print(f"\n  Step {step}:")
    print(f"    Predictions: {np.round(pred, 2)}")
    print(f"    Residuals:   {np.round(y_demo - pred, 2)}")
    print(f"    MSE:         {mse:.2f}")

print("\n  ✦ Notice: residuals get smaller at each step! That's boosting.")

# ── Section 3: Gradient Boosting Classifier ───────────────────────
print("\n" + "=" * 65)
print("SECTION 3: GradientBoostingClassifier (scikit-learn)")
print("=" * 65)

# --- 3.1 Basic usage ---
print("\n📌 3.1 Basic Gradient Boosting Classifier:")
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)
gb_clf.fit(X_train, y_train)

print(f"  Train Accuracy: {accuracy_score(y_train, gb_clf.predict(X_train)):.4f}")
print(f"  Test Accuracy:  {accuracy_score(y_test, gb_clf.predict(X_test)):.4f}")
print(
    f"  CV Accuracy:    {cross_val_score(gb_clf, X_train, y_train, cv=5).mean():.4f}"
)

# --- 3.2 Learning rate vs n_estimators trade-off ---
print("\n📌 3.2 Learning Rate vs n_estimators Trade-off:")
print("  (lower lr needs more estimators for same performance)\n")

configs = [
    (0.5, 50),
    (0.1, 100),
    (0.1, 200),
    (0.05, 200),
    (0.05, 500),
    (0.01, 500),
    (0.01, 1000),
]

for lr, n_est in configs:
    gb = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=lr, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"  lr={lr:.2f}, n_estimators={n_est:>4d}: test_acc={test_acc:.4f}")

# --- 3.3 Effect of max_depth ---
print("\n📌 3.3 Effect of max_depth:")
for depth in [1, 2, 3, 5, 7, 10]:
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=42
    )
    gb.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, gb.predict(X_train))
    test_acc = accuracy_score(y_test, gb.predict(X_test))
    gap = train_acc - test_acc
    print(
        f"  max_depth={depth:>2d}: train={train_acc:.4f}, "
        f"test={test_acc:.4f}, gap={gap:.4f} "
        f"{'⚠️ overfitting' if gap > 0.05 else '✅'}"
    )

# ── Section 4: Gradient Boosting Regressor ────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Gradient Boosting Regressor")
print("=" * 65)

X_reg, y_reg = make_regression(
    n_samples=500, n_features=10, n_informative=5, noise=20, random_state=42
)
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
)
gb_reg.fit(X_r_train, y_r_train)
y_pred = gb_reg.predict(X_r_test)

print(f"\n  R² Score: {r2_score(y_r_test, y_pred):.4f}")
print(f"  RMSE:     {np.sqrt(mean_squared_error(y_r_test, y_pred)):.4f}")

# Staged predictions (training curve)
print("\n  Training progress (staged R²):")
train_scores = []
test_scores = []
for y_pred_staged in gb_reg.staged_predict(X_r_test):
    test_scores.append(r2_score(y_r_test, y_pred_staged))

checkpoints = [0, 24, 49, 99, 149, 199]
for idx in checkpoints:
    if idx < len(test_scores):
        print(f"    After {idx + 1:>3d} rounds: R² = {test_scores[idx]:.4f}")

# --- Subsample for Stochastic Gradient Boosting ---
print("\n📌 Stochastic Gradient Boosting (subsample < 1.0):")
for subsample in [0.5, 0.7, 0.8, 1.0]:
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=subsample,
        random_state=42,
    )
    gb.fit(X_r_train, y_r_train)
    r2 = r2_score(y_r_test, gb.predict(X_r_test))
    print(f"  subsample={subsample:.1f}: R² = {r2:.4f}")

# ── Section 5: Bagging vs Boosting Comparison ─────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Bagging vs Boosting — Head-to-Head")
print("=" * 65)

# Use Wine dataset
wine = load_wine()
X_w, y_w = wine.data, wine.target
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X_w, y_w, test_size=0.2, random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}

print(f"\n  {'Model':<25s} {'Train':>8s} {'Test':>8s} {'CV(5)':>8s}")
print("  " + "-" * 51)

for name, model in models.items():
    model.fit(X_w_train, y_w_train)
    train_acc = accuracy_score(y_w_train, model.predict(X_w_train))
    test_acc = accuracy_score(y_w_test, model.predict(X_w_test))
    cv_acc = cross_val_score(model, X_w_train, y_w_train, cv=5).mean()
    print(f"  {name:<25s} {train_acc:>8.4f} {test_acc:>8.4f} {cv_acc:>8.4f}")

# ── Section 6: Feature Importance from Gradient Boosting ──────────
print("\n" + "=" * 65)
print("SECTION 6: Feature Importance (Gradient Boosting)")
print("=" * 65)

gb_wine = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)
gb_wine.fit(X_w_train, y_w_train)

importances = gb_wine.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\n  Top 5 Features (by impurity decrease):")
for i in range(5):
    idx = sorted_idx[i]
    print(f"    {i + 1}. {wine.feature_names[idx]:>25s}: {importances[idx]:.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(
    """
✅ Key Takeaways:
  1. AdaBoost: reweights misclassified samples
  2. Gradient Boosting: fits residuals sequentially
  3. Lower learning_rate → more estimators needed → better generalization
  4. max_depth 3-5 works well for Gradient Boosting (shallow trees!)
  5. subsample < 1.0 adds regularization (Stochastic GB)
  6. Boosting can overfit! Use early stopping or regularization
  7. Gradient Boosting often outperforms Random Forest on tabular data

📊 Comparison:
  ┌──────────────────┬─────────────────┬─────────────────┐
  │                  │    Bagging (RF)  │    Boosting      │
  ├──────────────────┼─────────────────┼─────────────────┤
  │ Strategy         │  Parallel        │  Sequential      │
  │ Reduces          │  Variance        │  Bias + Variance │
  │ Overfitting risk │  Low             │  Higher          │
  │ Parallelizable   │  Yes             │  Limited         │
  │ Typical depth    │  Deep            │  Shallow         │
  └──────────────────┴─────────────────┴─────────────────┘

📚 Next: 03_xgboost_advanced.py (XGBoost & LightGBM)
"""
)
