"""
=================================================================
02 - KERNELS: The Kernel Trick Deep Dive
=================================================================
Topics:
  1. Why kernels are needed
  2. Linear kernel
  3. RBF (Gaussian) kernel
  4. Polynomial kernel
  5. Gamma parameter analysis
  6. Kernel comparison on real data
=================================================================
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Section 1: Why Kernels? Non-linear Data ──────────────────────
print("=" * 65)
print("SECTION 1: Why Do We Need Kernels?")
print("=" * 65)

# Create non-linear dataset (moons shape)
X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=42)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42
)

# Linear kernel fails on non-linear data
svm_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0))
])
svm_linear.fit(X_m_train, y_m_train)

# RBF kernel handles non-linear data
svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0))
])
svm_rbf.fit(X_m_train, y_m_train)

print(f"\n  Moon-shaped data (non-linear):")
print(f"    Linear kernel accuracy: {accuracy_score(y_m_test, svm_linear.predict(X_m_test)):.4f}")
print(f"    RBF kernel accuracy:    {accuracy_score(y_m_test, svm_rbf.predict(X_m_test)):.4f}")

# Circles data
X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
    X_circles, y_circles, test_size=0.2, random_state=42
)

svm_lin_c = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="linear"))])
svm_rbf_c = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="rbf"))])
svm_lin_c.fit(X_c_train, y_c_train)
svm_rbf_c.fit(X_c_train, y_c_train)

print(f"\n  Concentric circles data:")
print(f"    Linear kernel accuracy: {accuracy_score(y_c_test, svm_lin_c.predict(X_c_test)):.4f}")
print(f"    RBF kernel accuracy:    {accuracy_score(y_c_test, svm_rbf_c.predict(X_c_test)):.4f}")

print("\n  💡 Linear kernel CANNOT separate non-linear data!")
print("     RBF kernel maps to higher dimensions where data IS separable.")

# ── Section 2: Linear Kernel ─────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Linear Kernel")
print("=" * 65)

print("""
  K(x, y) = x · y  (dot product)

  ✅ Use when:
    • Data is linearly separable (or close to it)
    • High-dimensional data (text, genomics)
    • Very large datasets (fastest kernel)
    • Number of features >> number of samples
""")

from sklearn.datasets import make_classification
X_lin, y_lin = make_classification(
    n_samples=500, n_features=50, n_informative=20,
    n_redundant=10, random_state=42
)
X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)

pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="linear", C=1.0))])
pipe.fit(X_l_train, y_l_train)
print(f"  High-dimensional data (50 features):")
print(f"  Linear kernel accuracy: {accuracy_score(y_l_test, pipe.predict(X_l_test)):.4f}")
print(f"  CV score: {cross_val_score(pipe, X_l_train, y_l_train, cv=5).mean():.4f}")

# ── Section 3: RBF Kernel ────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: RBF (Radial Basis Function) Kernel")
print("=" * 65)

print("""
  K(x, y) = exp(-γ × ||x - y||²)

  γ (gamma) controls the "reach" of each training example:
    • High γ → each point has small reach → complex boundary → overfitting
    • Low γ  → each point has wide reach → smooth boundary → underfitting
""")

# Show gamma effect
X_moons, y_moons = make_moons(n_samples=500, noise=0.3, random_state=42)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42
)

print(f"\n  {'gamma':>10s} {'Train Acc':>10s} {'Test Acc':>10s} {'# SVs':>8s} {'Status':>15s}")
print("  " + "-" * 58)

for gamma in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, gamma=gamma))])
    pipe.fit(X_m_train, y_m_train)
    tr_acc = accuracy_score(y_m_train, pipe.predict(X_m_train))
    te_acc = accuracy_score(y_m_test, pipe.predict(X_m_test))
    n_sv = sum(pipe.named_steps["svm"].n_support_)
    
    if te_acc < 0.75:
        status = "❌ Underfitting"
    elif tr_acc - te_acc > 0.1:
        status = "⚠️ Overfitting"
    else:
        status = "✅ Good"
    
    print(f"  {gamma:>10.3f} {tr_acc:>10.4f} {te_acc:>10.4f} {n_sv:>8d} {status:>15s}")

print("\n  💡 'scale' gamma = 1 / (n_features × X.var()) — good default!")

# ── Section 4: Polynomial Kernel ──────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Polynomial Kernel")
print("=" * 65)

print("""
  K(x, y) = (γ × x·y + r)^d

  Parameters:
    • degree (d): polynomial degree
    • gamma (γ): kernel coefficient
    • coef0 (r): independent term
""")

print(f"\n  Polynomial kernel on moon data:")
print(f"  {'degree':>8s} {'Train Acc':>10s} {'Test Acc':>10s}")
print("  " + "-" * 32)

for degree in [2, 3, 4, 5, 6]:
    pipe = Pipeline([
        ("s", StandardScaler()),
        ("svm", SVC(kernel="poly", degree=degree, C=1.0, coef0=1))
    ])
    pipe.fit(X_m_train, y_m_train)
    tr_acc = accuracy_score(y_m_train, pipe.predict(X_m_train))
    te_acc = accuracy_score(y_m_test, pipe.predict(X_m_test))
    print(f"  {degree:>8d} {tr_acc:>10.4f} {te_acc:>10.4f}")

print("\n  💡 Higher degree = more flexibility but slower and might overfit")

# ── Section 5: Kernel Comparison on Real Data ─────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Kernel Comparison — Wine Dataset")
print("=" * 65)

wine = load_wine()
X_w, y_w = wine.data, wine.target
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X_w, y_w, test_size=0.2, random_state=42
)

kernels = {
    "linear": {"kernel": "linear", "C": 1.0},
    "rbf (default)": {"kernel": "rbf", "C": 1.0},
    "rbf (tuned)": {"kernel": "rbf", "C": 10.0, "gamma": 0.01},
    "poly (d=2)": {"kernel": "poly", "degree": 2, "C": 1.0, "coef0": 1},
    "poly (d=3)": {"kernel": "poly", "degree": 3, "C": 1.0, "coef0": 1},
}

print(f"\n  {'Kernel':<16s} {'Test Acc':>10s} {'CV(5)':>10s}")
print("  " + "-" * 38)

for name, params in kernels.items():
    pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(**params))])
    pipe.fit(X_w_train, y_w_train)
    te_acc = accuracy_score(y_w_test, pipe.predict(X_w_test))
    cv_acc = cross_val_score(pipe, X_w_train, y_w_train, cv=5).mean()
    print(f"  {name:<16s} {te_acc:>10.4f} {cv_acc:>10.4f}")

# ── Section 6: Finding Best C and Gamma with GridSearch ───────────
print("\n" + "=" * 65)
print("SECTION 6: GridSearchCV for C and Gamma")
print("=" * 65)

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1],
}

pipe = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="rbf"))])
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_w_train, y_w_train)

print(f"\n  Best Parameters: {grid.best_params_}")
print(f"  Best CV Score:   {grid.best_score_:.4f}")
print(f"  Test Accuracy:   {accuracy_score(y_w_test, grid.predict(X_w_test)):.4f}")

# Show top 5 parameter combinations
print("\n  Top 5 parameter combinations:")
results = grid.cv_results_
sorted_idx = np.argsort(results["mean_test_score"])[::-1]
for i in range(5):
    idx = sorted_idx[i]
    print(f"    {i+1}. C={results['params'][idx]['svm__C']}, "
          f"gamma={results['params'][idx]['svm__gamma']}: "
          f"{results['mean_test_score'][idx]:.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. Linear kernel: fast, for linearly separable / high-dim data
  2. RBF kernel: most versatile, handles non-linear patterns
  3. Gamma controls boundary complexity (high=complex, low=smooth)
  4. Polynomial: good for polynomial relationships, degree matters
  5. Always use GridSearchCV to find best C and gamma together
  6. 'scale' gamma is a good starting point

🔑 Quick Decision Guide:
  • High-dim text data → Linear kernel
  • General purpose → RBF with gamma='scale'
  • Known polynomial relationship → Poly kernel
  • Not sure → Try RBF first, then compare

📚 Next: 03_multiclass_svm.py
""")
