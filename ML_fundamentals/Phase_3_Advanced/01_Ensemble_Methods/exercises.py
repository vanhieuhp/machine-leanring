"""
Ensemble Methods - Exercises
============================

Practice problems to reinforce your understanding of ensemble methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.datasets import make_classification, make_regression, load_iris, load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not installed")

# ============================================================================
# EXERCISE 1: Random Forest Basics
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Random Forest Basics")
print("=" * 70)

print("""
Task:
1. Load the breast cancer dataset
2. Split into train/test (80/20)
3. Train a Random Forest with n_estimators=100
4. Evaluate accuracy
5. Find the most important feature
""")

# Your code here:
# 1. Load breast cancer dataset
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

# 3. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 4. Evaluate
accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"\nAccuracy: {accuracy:.4f}")

# 5. Most important feature
feature_importance = list(zip(cancer.feature_names, rf.feature_importances_))
feature_importance.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Most Important Features:")
for name, importance in feature_importance[:5]:
    print(f"  {name}: {importance:.4f}")

# ============================================================================
# EXERCISE 2: Compare Random Forest vs Gradient Boosting
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Compare Random Forest vs Gradient Boosting")
print("=" * 70)

print("""
Task:
1. Generate classification data (n_samples=1000, n_features=10)
2. Train Random Forest (n_estimators=100)
3. Train Gradient Boosting (n_estimators=100)
4. Compare accuracies
5. Plot learning curves (accuracy vs n_estimators)
""")

# Your code here:
# 1. Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# 3. Train Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))

# 4. Compare
print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

# 5. Learning curves
print("\nGenerating learning curves...")
n_estimators_range = [10, 25, 50, 75, 100, 150, 200]
rf_scores = []
gb_scores = []

for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train, y_train)
    rf_scores.append(rf_temp.score(X_test, y_test))

    gb_temp = GradientBoostingClassifier(n_estimators=n, random_state=42)
    gb_temp.fit(X_train, y_train)
    gb_scores.append(gb_temp.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, rf_scores, 'o-', label='Random Forest')
plt.plot(n_estimators_range, gb_scores, 's-', label='Gradient Boosting')
plt.xlabel('n_estimators')
plt.ylabel('Test Accuracy')
plt.title('Learning Curve: Random Forest vs Gradient Boosting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# EXERCISE 3: XGBoost Hyperparameter Tuning
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: XGBoost Hyperparameter Tuning")
print("=" * 70)

if XGB_AVAILABLE:
    print("""
Task:
1. Use breast cancer dataset
2. Perform grid search for:
   - max_depth: [3, 5, 7]
   - learning_rate: [0.05, 0.1, 0.2]
3. Use 3-fold cross-validation
4. Report best parameters and score
""")

    # Your code here:
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2]
    }

    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)

    grid_search = GridSearchCV(
        xgb_clf,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print(f"Test Score: {grid_search.best_estimator_.score(X_test, y_test):.4f}")
else:
    print("Install XGBoost: pip install xgboost")

# ============================================================================
# EXERCISE 4: Build Voting Ensemble
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Build Voting Ensemble")
print("=" * 70)

print("""
Task:
1. Load iris dataset
2. Create ensemble with:
   - Random Forest
   - Logistic Regression
   - Decision Tree
3. Compare Hard Voting vs Soft Voting
""")

# Your code here:
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Create classifiers
rf = RandomForestClassifier(n_estimators=50, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)

# Hard Voting
hard_voting = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('dt', dt)],
    voting='hard'
)

# Soft Voting
soft_voting = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('dt', dt)],
    voting='soft'
)

# Train and evaluate
hard_voting.fit(X_train, y_train)
soft_voting.fit(X_train, y_train)

hard_acc = accuracy_score(y_test, hard_voting.predict(X_test))
soft_acc = accuracy_score(y_test, soft_voting.predict(X_test))

print(f"\nHard Voting Accuracy: {hard_acc:.4f}")
print(f"Soft Voting Accuracy: {soft_acc:.4f}")

# ============================================================================
# EXERCISE 5: Handle Imbalanced Data
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Handle Imbalanced Data with Ensemble")
print("=" * 70)

print("""
Task:
1. Create imbalanced dataset (10% positive class)
2. Train Random Forest without class_weight
3. Train Random Forest with class_weight='balanced'
4. Compare F1 scores
""")

# Your code here:
# 1. Create imbalanced data
X_imbalanced, y_imbalanced = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imbalanced, y_imbalanced, test_size=0.2, random_state=42
)

# 2. Train without class_weight
rf_normal = RandomForestClassifier(n_estimators=100, random_state=42)
rf_normal.fit(X_train, y_train)
y_pred_normal = rf_normal.predict(X_test)

# 3. Train with class_weight
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

# 4. Compare
from sklearn.metrics import f1_score, precision_score, recall_score

print(f"\nWithout class_weight:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_normal):.4f}")
print(f"  F1 Score: {f1_score(y_test, y_pred_normal):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_normal):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_normal):.4f}")

print(f"\nWith class_weight='balanced':")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_balanced):.4f}")
print(f"  F1 Score: {f1_score(y_test, y_pred_balanced):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_balanced):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_balanced):.4f}")

# ============================================================================
# EXERCISE 6: Build Stacking Ensemble
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Build Stacking Ensemble")
print("=" * 70)

print("""
Task:
1. Use breast cancer dataset
2. Create stacking ensemble with:
   - Base models: RF, GB, NB
   - Meta-learner: Logistic Regression
3. Evaluate and compare with individual models
""")

# Your code here:
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Individual models
rf = RandomForestClassifier(n_estimators=50, random_state=42)
gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

# Train individual models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
nb.fit(X_train, y_train)

print("\nIndividual Model Scores:")
print(f"  Random Forest: {rf.score(X_test, y_test):.4f}")
print(f"  Gradient Boosting: {gb.score(X_test, y_test):.4f}")
print(f"  Naive Bayes: {nb.score(X_test, y_test):.4f}")

# Stacking
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('nb', GaussianNB())
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking.fit(X_train, y_train)
print(f"\nStacking Score: {stacking.score(X_test, y_test):.4f}")

# ============================================================================
# EXERCISE 7: LightGBM (if available)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: LightGBM")
print("=" * 70)

if LGB_AVAILABLE:
    print("""
Task:
1. Use breast cancer dataset
2. Train LightGBM classifier
3. Use early stopping
4. Evaluate performance
""")

    # Your code here:
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )

    lgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )

    print(f"\nBest Iteration: {lgb_clf.best_iteration_}")
    print(f"Test Accuracy: {lgb_clf.score(X_test, y_test):.4f}")
else:
    print("Install LightGBM: pip install lightgbm")

# ============================================================================
# BONUS EXERCISE: Feature Engineering with Ensemble
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: Feature Engineering with Ensemble")
print("=" * 70)

print("""
Task:
1. Generate regression data with noise
2. Create polynomial features (degree 2)
3. Train Random Forest Regressor
4. Evaluate R² score
""")

# Your code here:
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

# 1. Generate data
X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 2. Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features: {X_train_poly.shape[1]}")

# 3. Train Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_poly, y_train)
y_pred = rf_reg.predict(X_test_poly)

# 4. Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nR² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
What you practiced:
1. Random Forest classification
2. Random Forest vs Gradient Boosting comparison
3. XGBoost hyperparameter tuning
4. Voting ensemble (Hard vs Soft)
5. Handling imbalanced data
6. Stacking ensemble
7. LightGBM with early stopping
8. Feature engineering with ensemble

Next Steps:
- Try these on real datasets
- Experiment with different hyperparameters
- Combine multiple techniques
""")
