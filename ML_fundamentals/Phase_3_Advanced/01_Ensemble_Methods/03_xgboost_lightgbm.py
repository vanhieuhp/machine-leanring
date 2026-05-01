"""
XGBoost & LightGBM - Advanced Gradient Boosting
================================================

This module covers:
- XGBoost: Extreme Gradient Boosting
- LightGBM: Light Gradient Boosting Machine
- Key differences and advantages
- Hyperparameter tuning
- Early stopping
- Cross-validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.datasets import make_classification, make_regression

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("XGBoost version:", xgb.__version__)
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print("LightGBM version:", lgb.__version__)
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not installed. Install with: pip install lightgbm")

# ============================================================================
# 1. XGBOOST OVERVIEW
# ============================================================================

print("\n" + "=" * 70)
print("1. XGBOOST OVERVIEW")
print("=" * 70)

print("""
XGBoost (Extreme Gradient Boosting):
- Optimized implementation of gradient boosting
- Used extensively in Kaggle competitions

Key Features:
1. Regularization: L1 (Lasso) and L2 (Ridge) on leaf weights
2. Parallel Learning: Block structure for parallel computation
3. Cache-aware Access: Optimized for CPU cache
4. Missing Values: Built-in handling
5. Early Stopping: Stop when validation doesn't improve
6. Built-in Cross-Validation

Regularization in XGBoost:
- reg_alpha: L1 regularization (sparse features)
- reg_lambda: L2 regularization (smooth weights)
- Both prevent overfitting
""")

# ============================================================================
# 2. XGBOOST CLASSIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("2. XGBOOST CLASSIFICATION")
print("=" * 70)

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                            n_redundant=2, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if XGB_AVAILABLE:
    # Create XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,  # L1 regularization
        reg_lambda=1,  # L2 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)

    print("\nXGBoost Classifier Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    plt.figure(figsize=(10, 5))
    plt.barh([f'Feature {i}' for i in range(10)], xgb_classifier.feature_importances_, color='green')
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()
else:
    print("Install XGBoost to see results")

# ============================================================================
# 3. XGBOOST REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("3. XGBOOST REGRESSION")
print("=" * 70)

if XGB_AVAILABLE:
    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    xgb_regressor = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = xgb_regressor.predict(X_test_reg)

    print("\nXGBoost Regressor Results:")
    print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")

# ============================================================================
# 4. EARLY STOPPING WITH XGBOOST
# ============================================================================

print("\n" + "=" * 70)
print("4. EARLY STOPPING")
print("=" * 70)

print("""
Early Stopping:
- Monitor validation error during training
- Stop when no improvement for n rounds
- Finds optimal number of boosting rounds
- Prevents overfitting
""")

if XGB_AVAILABLE:
    # Split data for early stopping
    X_train_es, X_val, y_train_es, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    xgb_early = xgb.XGBClassifier(
        n_estimators=500,  # More than needed
        max_depth=6,
        learning_rate=0.1,
        early_stopping_rounds=10,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    xgb_early.fit(X_train_es, y_train_es, eval_set=[(X_val, y_val)], verbose=False)

    print(f"Best iteration: {xgb_early.best_iteration}")
    print(f"Best score: {xgb_early.best_score:.4f}")
    print(f"Test accuracy: {xgb_early.score(X_test, y_test):.4f}")

# ============================================================================
# 5. LIGHTGBM OVERVIEW
# ============================================================================

print("\n" + "=" * 70)
print("5. LIGHTGBM OVERVIEW")
print("=" * 70)

print("""
LightGBM (Light Gradient Boosting Machine):
- Microsoft's gradient boosting framework
- Designed for speed and efficiency

Key Features:
1. Histogram-based: Bins continuous features (faster)
2. GOSS: Gradient-based One-Side Sampling
   - Focus on instances with large gradients
   - Random sample from small-gradient instances
3. EFB: Exclusive Feature Bundling
   - Bundle mutually exclusive features
   - Reduces feature count

Why LightGBM is Faster:
- Lower memory usage (histogram)
- Faster training (GOSS, EFB)
- Handles large datasets well
- Leaf-wise tree growth (vs level-wise)
""")

# ============================================================================
# 6. LIGHTGBM CLASSIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("6. LIGHTGBM CLASSIFICATION")
print("=" * 70)

if LGB_AVAILABLE:
    lgb_classifier = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,  # Max leaves per tree
        max_depth=-1,   # No limit
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=0,
        random_state=42,
        verbose=-1
    )

    lgb_classifier.fit(X_train, y_train)
    y_pred_lgb = lgb_classifier.predict(X_test)

    print("\nLightGBM Classifier Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")

    # Feature importance
    plt.figure(figsize=(10, 5))
    plt.barh([f'Feature {i}' for i in range(10)], lgb_classifier.feature_importances_, color='blue')
    plt.xlabel('Importance')
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.show()
else:
    print("Install LightGBM to see results")

# ============================================================================
# 7. KEY DIFFERENCES: XGBOOST vs LIGHTGBM
# ============================================================================

print("\n" + "=" * 70)
print("7. XGBOOST vs LIGHTGBM")
print("=" * 70)

print("""
| Aspect           | XGBoost          | LightGBM         |
|------------------|------------------|------------------|
| Tree Growth      | Level-wise       | Leaf-wise        |
| Split Finding    | Exact/Histogram  | Histogram only   |
| Speed            | Medium           | Very Fast        |
| Memory           | Medium           | Low              |
| Accuracy         | Excellent        | Excellent        |
| Large Datasets   | Good             | Excellent        |
| Categorical      | Native support   | Native support   |

Note: Leaf-wise grows trees faster but may overfit on small datasets.
""")

# ============================================================================
# 8. HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "=" * 70)
print("8. HYPERPARAMETER TUNING")
print("=" * 70)

print("""
XGBoost Key Parameters:
- n_estimators: Number of trees
- max_depth: Max depth per tree
- learning_rate: Step size
- subsample: Row sampling
- colsample_bytree: Feature sampling
- reg_alpha: L1 regularization
- reg_lambda: L2 regularization
- min_child_weight: Min sum of instance weight

LightGBM Key Parameters:
- n_estimators: Number of trees
- num_leaves: Max leaves per tree
- max_depth: Max depth (limit)
- learning_rate: Step size
- subsample: Row sampling
- colsample_bytree: Feature sampling
- reg_alpha: L1 regularization
- reg_lambda: L2 regularization
- min_child_samples: Min samples in leaf

Tuning Strategy:
1. Start with default parameters
2. Tune max_depth/num_leaves
3. Tune learning_rate + n_estimators
4. Tune subsample/colsample
5. Tune regularization
""")

# ============================================================================
# 9. PRACTICAL TUNING EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL TUNING EXAMPLE")
print("=" * 70)

if XGB_AVAILABLE:
    # Define parameter grid (small for demo)
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [50, 100]
    }

    # Grid search (use smaller grid for speed)
    xgb_tune = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    grid_search = GridSearchCV(
        xgb_tune,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Test on held-out data
    y_pred_tuned = grid_search.best_estimator_.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

# ============================================================================
# 10. CROSS-VALIDATION WITH XGBOOST
# ============================================================================

print("\n" + "=" * 70)
print("10. CROSS-VALIDATION")
print("=" * 70)

if XGB_AVAILABLE:
    # Using XGBoost's built-in CV
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=100,
        nfold=5,
        metrics=['error'],
        early_stopping_rounds=10,
        seed=42,
        verbose_eval=False
    )

    print("\nXGBoost Cross-Validation Results:")
    print(f"Best iteration: {len(cv_results)}")
    print(f"Best CV error: {cv_results['test-error-mean'].min():.4f}")

    # Plot CV results
    plt.figure(figsize=(10, 5))
    plt.plot(cv_results['test-error-mean'], label='Test Error')
    plt.fill_between(
        range(len(cv_results)),
        cv_results['test-error-mean'] - cv_results['test-error-std'],
        cv_results['test-error-mean'] + cv_results['test-error-std'],
        alpha=0.3
    )
    plt.xlabel('Boosting Round')
    plt.ylabel('Error')
    plt.title('XGBoost CV Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================================
# 11. SPEED COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("11. SPEED COMPARISON")
print("=" * 70)

from sklearn.ensemble import GradientBoostingClassifier
import time

# Large dataset
X_large, y_large = make_classification(n_samples=5000, n_features=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

models = []

# sklearn GradientBoosting
gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
models.append(('sklearn GB', gb))

# XGBoost
if XGB_AVAILABLE:
    xgb_m = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42, verbosity=0)
    models.append(('XGBoost', xgb_m))

# LightGBM
if LGB_AVAILABLE:
    lgb_m = lgb.LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
    models.append(('LightGBM', lgb_m))

print("\nTraining Time Comparison (n_samples=5000, n_estimators=50):")
print("-" * 40)

times = []
for name, model in models:
    start = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - start
    accuracy = model.score(X_te, y_te)
    print(f"{name:15s}: {elapsed:.2f}s, Accuracy: {accuracy:.4f}")
    times.append(elapsed)

# ============================================================================
# 12. USING BOTH: XGBOOST + LIGHTGBM
# ============================================================================

print("\n" + "=" * 70)
print("12. USING BOTH XGBOOST AND LIGHTGBM")
print("=" * 70)

print("""
When to use which:

XGBoost:
- Smaller datasets
- When you need more control
- When regularization is important
- Kaggle competitions

LightGBM:
- Very large datasets (millions of rows)
- When speed is critical
- When memory is limited
- Real-time applications

Best Practice:
- Try both on your data
- Use the one that performs better
- Can combine in ensembles
""")

if XGB_AVAILABLE and LGB_AVAILABLE:
    # Train both
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=42, verbosity=0)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                                   random_state=42, verbose=-1)

    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)

    # Get predictions
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]

    # Average (ensemble)
    ensemble_pred = (xgb_pred + lgb_pred) / 2
    ensemble_class = (ensemble_pred > 0.5).astype(int)

    print("\nComparison:")
    print(f"XGBoost:   {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")
    print(f"LightGBM:  {accuracy_score(y_test, lgb_model.predict(X_test)):.4f}")
    print(f"Ensemble:  {accuracy_score(y_test, ensemble_class):.4f}")

print("\n" + "=" * 70)
print("XGBOOST & LIGHTGBM SUMMARY")
print("=" * 70)
print("""
Key Takeaways:
1. XGBoost: Regularized gradient boosting, great for competitions
2. LightGBM: Fastest for large datasets, leaf-wise growth
3. Both are faster than sklearn's GradientBoosting
4. Use early_stopping to prevent overfitting
5. Cross-validate to find optimal parameters
6. Try both and ensemble if needed

Installation:
- XGBoost: pip install xgboost
- LightGBM: pip install lightgbm

Next Steps:
- Learn stacking and voting ensembles
- Apply to real-world problems
""")
