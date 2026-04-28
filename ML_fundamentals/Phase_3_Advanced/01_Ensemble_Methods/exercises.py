"""
=================================================================
ENSEMBLE METHODS — EXERCISES
=================================================================
5 hands-on exercises with increasing difficulty.
Each exercise has: Description, Starter Code, Hints, and Solution.

Run each exercise independently. Try to solve before looking
at the solution!
=================================================================
"""

import numpy as np
from sklearn.datasets import (
    load_wine, load_breast_cancer, make_classification, make_regression
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score
import warnings
warnings.filterwarnings("ignore")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 1: Random Forest Basics (⭐⭐)                       ║
# ╚═════════════════════════════════════════════════════════════════╝
print("=" * 65)
print("EXERCISE 1: Build & Evaluate a Random Forest")
print("=" * 65)
print("""
📝 Task:
  1. Load the breast cancer dataset
  2. Split 80/20 train/test
  3. Train a RandomForestClassifier with n_estimators=200
  4. Print train accuracy, test accuracy, and OOB score
  5. Print the top 3 most important features

🎯 Expected: Test accuracy > 0.95
""")

# === YOUR CODE HERE ===
# from sklearn.ensemble import RandomForestClassifier
# data = load_breast_cancer()
# ...
# =====================


# --- SOLUTION (scroll down) ---
#
#
#
#
#
#
#
#
#
#
#
#
#
# === SOLUTION ===
print("--- SOLUTION ---")
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print(f"  Train Accuracy: {accuracy_score(y_train, rf.predict(X_train)):.4f}")
print(f"  Test Accuracy:  {accuracy_score(y_test, rf.predict(X_test)):.4f}")
print(f"  OOB Score:      {rf.oob_score_:.4f}")

print("\n  Top 3 features:")
sorted_idx = np.argsort(rf.feature_importances_)[::-1]
for i in range(3):
    idx = sorted_idx[i]
    print(f"    {i+1}. {data.feature_names[idx]}: {rf.feature_importances_[idx]:.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 2: Bagging vs Boosting Comparison (⭐⭐)              ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 2: Bagging vs Boosting Comparison")
print("=" * 65)
print("""
📝 Task:
  1. Create a synthetic dataset with make_classification
     (n_samples=1000, n_features=20, n_informative=10)
  2. Train 4 models: DecisionTree, RandomForest, AdaBoost, GradientBoosting
  3. Compare using 5-fold cross-validation
  4. Print results in a table format

🎯 Expected: Gradient Boosting should be the best
""")

# === YOUR CODE HERE ===
# ...
# =====================


# --- SOLUTION ---
#
#
#
#
#
#
#
#
#
#
#
# === SOLUTION ===
print("--- SOLUTION ---")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier
)

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print(f"\n  {'Model':<20s} {'Mean CV':>8s} {'Std':>8s}")
print("  " + "-" * 38)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"  {name:<20s} {scores.mean():>8.4f} {scores.std():>8.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 3: Hyperparameter Tuning (⭐⭐⭐)                     ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 3: Tune a Gradient Boosting Classifier")
print("=" * 65)
print("""
📝 Task:
  1. Use the wine dataset
  2. Use RandomizedSearchCV to tune GradientBoostingClassifier
     Parameters to tune:
       - n_estimators: [50, 100, 200, 300]
       - learning_rate: [0.01, 0.05, 0.1, 0.2]
       - max_depth: [2, 3, 4, 5]
       - subsample: [0.7, 0.8, 0.9, 1.0]
  3. Use n_iter=20, cv=5
  4. Print the best parameters and best score
  5. Evaluate on test set

🎯 Expected: Test accuracy > 0.94
""")

# === YOUR CODE HERE ===
# ...
# =====================


# --- SOLUTION ---
#
#
#
#
#
#
#
#
#
#
#
# === SOLUTION ===
print("--- SOLUTION ---")
from sklearn.model_selection import RandomizedSearchCV

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
}

rs = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist, n_iter=20, cv=5, scoring="accuracy",
    random_state=42, n_jobs=-1,
)
rs.fit(X_train, y_train)

print(f"  Best CV Score: {rs.best_score_:.4f}")
print(f"  Best Params:   {rs.best_params_}")
print(f"  Test Accuracy: {accuracy_score(y_test, rs.predict(X_test)):.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 4: Voting Ensemble (⭐⭐⭐)                           ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 4: Build a Voting Ensemble")
print("=" * 65)
print("""
📝 Task:
  1. Use the wine dataset
  2. Create a VotingClassifier with:
     - RandomForest (100 trees)
     - GradientBoosting (100 estimators)
     - SVM with RBF kernel (probability=True)
  3. Compare hard voting vs soft voting
  4. Also compare with individual model performances
  5. Which approach works best?

🎯 Expected: Soft voting should be >= best individual model
""")

# === YOUR CODE HERE ===
# ...
# =====================


# --- SOLUTION ---
#
#
#
#
#
#
#
#
#
#
#
# === SOLUTION ===
print("--- SOLUTION ---")
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base estimators
rf_est = RandomForestClassifier(n_estimators=100, random_state=42)
gb_est = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm_est = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))])

# Hard Voting
hard_vote = VotingClassifier(
    estimators=[("rf", rf_est), ("gb", gb_est), ("svm", svm_est)],
    voting="hard"
)
hard_vote.fit(X_train, y_train)

# Soft Voting
soft_vote = VotingClassifier(
    estimators=[("rf", rf_est), ("gb", gb_est), ("svm", svm_est)],
    voting="soft"
)
soft_vote.fit(X_train, y_train)

# Individual models
for name, model in [("RF", rf_est), ("GB", gb_est), ("SVM", svm_est)]:
    model.fit(X_train, y_train)
    print(f"  {name:>12s}: {accuracy_score(y_test, model.predict(X_test)):.4f}")

print(f"  {'Hard Voting':>12s}: {accuracy_score(y_test, hard_vote.predict(X_test)):.4f}")
print(f"  {'Soft Voting':>12s}: {accuracy_score(y_test, soft_vote.predict(X_test)):.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 5: Full Stacking Pipeline (⭐⭐⭐⭐)                   ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 5: Build a Complete Stacking Pipeline")
print("=" * 65)
print("""
📝 Task:
  1. Create a synthetic dataset (n_samples=2000, n_features=30)
  2. Build a StackingClassifier with:
     Base models:
       - RandomForest
       - GradientBoosting
       - KNN (n_neighbors=7)
       - SVM (with scaling pipeline)
     Meta-learner:
       - LogisticRegression
  3. Compare stacking vs best individual model
  4. Use 5-fold cross-validation for evaluation
  5. Print a comprehensive comparison table

🎯 Challenge: Beat each individual model with stacking
""")

# === YOUR CODE HERE ===
# ...
# =====================


# --- SOLUTION ---
#
#
#
#
#
#
#
#
#
#
#
# === SOLUTION ===
print("--- SOLUTION ---")
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples=2000, n_features=30, n_informative=15,
    n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=7)),
    ("svm", Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))])),
]

# Build stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
)
stacking.fit(X_train, y_train)

print(f"\n  {'Model':<20s} {'Test Acc':>10s} {'CV Mean':>10s}")
print("  " + "-" * 42)

for name, model in base_models:
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  {name:<20s} {test_acc:>10.4f} {cv_scores.mean():>10.4f}")

stacking_acc = accuracy_score(y_test, stacking.predict(X_test))
stacking_cv = cross_val_score(stacking, X_train, y_train, cv=3).mean()
print(f"  {'STACKING':<20s} {stacking_acc:>10.4f} {stacking_cv:>10.4f}")

print("\n✅ Exercises complete! Move on to 02_SVM next.")
