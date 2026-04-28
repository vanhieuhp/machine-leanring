"""
=================================================================
SVM — EXERCISES
=================================================================
5 hands-on exercises with increasing difficulty.
=================================================================
"""
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 1: Basic SVM with Scaling Pipeline (⭐⭐)             ║
# ╚═════════════════════════════════════════════════════════════════╝
print("=" * 65)
print("EXERCISE 1: SVM with Proper Scaling")
print("=" * 65)
print("""
📝 Task:
  1. Load breast cancer dataset
  2. Create a Pipeline: StandardScaler + SVC(kernel='rbf')
  3. Compare accuracy WITH vs WITHOUT scaling
  4. Print both results

🎯 Expected: Scaled model should be significantly better
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Without scaling
svm_noscale = SVC(kernel="rbf", C=1.0, random_state=42)
svm_noscale.fit(X_train, y_train)
acc_noscale = accuracy_score(y_test, svm_noscale.predict(X_test))

# With scaling pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, random_state=42))])
pipe.fit(X_train, y_train)
acc_scaled = accuracy_score(y_test, pipe.predict(X_test))

print(f"  Without scaling: {acc_noscale:.4f}")
print(f"  With scaling:    {acc_scaled:.4f}")
print(f"  Improvement:     {(acc_scaled - acc_noscale)*100:.1f}%")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 2: Kernel Comparison (⭐⭐)                           ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 2: Compare Kernels on Non-Linear Data")
print("=" * 65)
print("""
📝 Task:
  1. Generate moon-shaped data (make_moons, n=500, noise=0.3)
  2. Train SVM with each kernel: linear, rbf, poly(d=3)
  3. Use Pipeline with StandardScaler
  4. Compare test accuracy and CV scores

🎯 Expected: RBF should perform best on non-linear data
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernels = {"linear": {}, "rbf": {}, "poly": {"degree": 3, "coef0": 1}}
print(f"  {'Kernel':<10s} {'Test Acc':>10s} {'CV(5)':>10s}")
print("  " + "-" * 32)

for kernel, params in kernels.items():
    pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel=kernel, C=1.0, **params))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    cv = cross_val_score(pipe, X_train, y_train, cv=5).mean()
    print(f"  {kernel:<10s} {acc:>10.4f} {cv:>10.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 3: Hyperparameter Tuning (⭐⭐⭐)                     ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 3: GridSearchCV for Best C and Gamma")
print("=" * 65)
print("""
📝 Task:
  1. Use wine dataset
  2. Create Pipeline: StandardScaler + SVC(kernel='rbf')
  3. GridSearchCV with:
     C: [0.1, 1, 10, 100]
     gamma: ['scale', 0.001, 0.01, 0.1]
  4. Print best params, best CV score, and test accuracy

🎯 Expected: Test accuracy > 0.97
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="rbf"))])
param_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.001, 0.01, 0.1]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"  Best params: {grid.best_params_}")
print(f"  Best CV:     {grid.best_score_:.4f}")
print(f"  Test Acc:    {accuracy_score(y_test, grid.predict(X_test)):.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 4: Multi-class with Confusion Matrix (⭐⭐⭐)         ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 4: Multi-class SVM with Detailed Evaluation")
print("=" * 65)
print("""
📝 Task:
  1. Use iris dataset
  2. Train SVM with RBF kernel (use Pipeline)
  3. Print classification report
  4. Print accuracy for each class separately

🎯 Expected: All per-class accuracy > 0.90
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, random_state=42))])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"  Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 5: SVR Regression (⭐⭐⭐)                            ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 5: SVR for Regression")
print("=" * 65)
print("""
📝 Task:
  1. Load diabetes dataset
  2. Build SVR pipeline (StandardScaler + SVR)
  3. Tune C, gamma, and epsilon with GridSearchCV
  4. Print best params and R² score
  5. Compare with LinearRegression baseline

🎯 Expected: SVR R² > 0.40
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_r2 = r2_score(y_test, lr.predict(X_test))

# SVR with tuning
pipe = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="rbf"))])
param_grid = {
    "svr__C": [1, 10, 100],
    "svr__gamma": ["scale", 0.01, 0.1],
    "svr__epsilon": [0.01, 0.1, 0.5],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

svr_r2 = r2_score(y_test, grid.predict(X_test))

print(f"  Linear Regression R²: {lr_r2:.4f}")
print(f"  SVR (tuned) R²:       {svr_r2:.4f}")
print(f"  Best params:          {grid.best_params_}")

print("\n✅ SVM exercises complete! Move on to 03_Neural_Networks next.")
