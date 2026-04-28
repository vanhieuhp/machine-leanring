"""
=================================================================
03 - MULTI-CLASS SVM: OvO, OvR, and Practical Strategies
=================================================================
Topics:
  1. One-vs-One (OvO) strategy
  2. One-vs-Rest (OvR) strategy
  3. Comparison of strategies
  4. Probability calibration
  5. Real-world multi-class example
=================================================================
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Section 1: One-vs-One (OvO) ──────────────────────────────────
print("=" * 65)
print("SECTION 1: One-vs-One (OvO) Strategy")
print("=" * 65)

print("""
  For K classes, train K(K-1)/2 binary classifiers.
  Each classifier is trained on 2 classes only.
  Final prediction: majority vote among all classifiers.

  Example (3 classes: A, B, C):
    Classifier 1: A vs B
    Classifier 2: A vs C
    Classifier 3: B vs C
    → 3 classifiers total
""")

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVC uses OvO by default
svm_ovo = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, decision_function_shape="ovo"))
])
svm_ovo.fit(X_train, y_train)

n_classifiers = 3 * 2 // 2  # K(K-1)/2
print(f"\n  Iris: {len(iris.target_names)} classes → {n_classifiers} binary classifiers")
print(f"  OvO Accuracy: {accuracy_score(y_test, svm_ovo.predict(X_test)):.4f}")

# ── Section 2: One-vs-Rest (OvR) ─────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: One-vs-Rest (OvR) Strategy")
print("=" * 65)

print("""
  For K classes, train K binary classifiers.
  Each classifier: "this class" vs "all other classes".
  Final prediction: class with highest confidence score.

  Example (3 classes: A, B, C):
    Classifier 1: A vs (B + C)
    Classifier 2: B vs (A + C)
    Classifier 3: C vs (A + B)
    → 3 classifiers total
""")

# LinearSVC uses OvR by default
svm_ovr = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=5000))
])
svm_ovr.fit(X_train, y_train)

print(f"  Iris: {len(iris.target_names)} classes → {len(iris.target_names)} binary classifiers")
print(f"  OvR Accuracy: {accuracy_score(y_test, svm_ovr.predict(X_test)):.4f}")

# Explicit OvR wrapper
svm_explicit_ovr = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", OneVsRestClassifier(SVC(kernel="rbf", C=1.0)))
])
svm_explicit_ovr.fit(X_train, y_train)
print(f"  Explicit OvR (RBF): {accuracy_score(y_test, svm_explicit_ovr.predict(X_test)):.4f}")

# ── Section 3: OvO vs OvR Comparison ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: OvO vs OvR Comparison")
print("=" * 65)

print("""
  ┌──────────────┬──────────────────────┬──────────────────────┐
  │              │ One-vs-One (OvO)     │ One-vs-Rest (OvR)    │
  ├──────────────┼──────────────────────┼──────────────────────┤
  │ # classifiers│ K(K-1)/2             │ K                    │
  │ Train data   │ Subset (2 classes)   │ All data             │
  │ Speed (train)│ Faster (small data)  │ Faster (large data)  │
  │ Speed (pred) │ Slower               │ Faster               │
  │ sklearn SVC  │ ✅ Default           │ Set explicitly       │
  │ LinearSVC    │ Set explicitly       │ ✅ Default           │
  └──────────────┴──────────────────────┴──────────────────────┘
""")

# Compare on Wine dataset (3 classes)
wine = load_wine()
X_w, y_w = wine.data, wine.target
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X_w, y_w, test_size=0.2, random_state=42
)

strategies = {
    "SVC (OvO default)": Pipeline([("s", StandardScaler()), ("svm", SVC(C=1.0))]),
    "LinearSVC (OvR)": Pipeline([("s", StandardScaler()), ("svm", LinearSVC(C=1.0, max_iter=5000))]),
    "OvR(SVC rbf)": Pipeline([("s", StandardScaler()), ("svm", OneVsRestClassifier(SVC(C=1.0)))]),
    "OvO(LinearSVC)": Pipeline([("s", StandardScaler()), ("svm", OneVsOneClassifier(LinearSVC(C=1.0, max_iter=5000)))]),
}

print(f"  Wine Dataset ({len(wine.target_names)} classes):")
print(f"  {'Strategy':<22s} {'Test Acc':>10s} {'CV(5)':>10s}")
print("  " + "-" * 44)

for name, model in strategies.items():
    model.fit(X_w_train, y_w_train)
    te_acc = accuracy_score(y_w_test, model.predict(X_w_test))
    cv_acc = cross_val_score(model, X_w_train, y_w_train, cv=5).mean()
    print(f"  {name:<22s} {te_acc:>10.4f} {cv_acc:>10.4f}")

# ── Section 4: Probability Estimates ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Probability Estimates")
print("=" * 65)

print("""
  SVM doesn't natively output probabilities.
  probability=True uses Platt scaling (sigmoid calibration):
    P(class|x) ≈ 1 / (1 + exp(A × f(x) + B))
  
  ⚠️ Warning: This adds training time (internal 5-fold CV)
""")

svm_prob = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
])
svm_prob.fit(X_w_train, y_w_train)

# Show probabilities for first 5 test samples
probs = svm_prob.predict_proba(X_w_test[:5])
preds = svm_prob.predict(X_w_test[:5])

print(f"\n  Sample predictions with probabilities:")
print(f"  {'Sample':>8s} {'Pred':>6s} {'P(0)':>8s} {'P(1)':>8s} {'P(2)':>8s} {'Confidence':>12s}")
print("  " + "-" * 48)
for i in range(5):
    confidence = probs[i].max()
    print(f"  {i:>8d} {preds[i]:>6d} {probs[i][0]:>8.3f} {probs[i][1]:>8.3f} {probs[i][2]:>8.3f} {confidence:>12.3f}")

# ── Section 5: Digits Classification ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Real-World Example — Digit Recognition")
print("=" * 65)

digits = load_digits()
X_d, y_d = digits.data, digits.target
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

print(f"\n  Dataset: {X_d.shape[0]} samples, {X_d.shape[1]} features, {len(np.unique(y_d))} classes")
print(f"  Classes: digits 0-9")

# Compare kernels
import time
kernels = {
    "Linear": SVC(kernel="linear", C=1.0),
    "RBF": SVC(kernel="rbf", C=10.0, gamma="scale"),
    "Poly(d=3)": SVC(kernel="poly", degree=3, C=1.0, coef0=1),
}

print(f"\n  {'Kernel':<12s} {'Accuracy':>10s} {'Time':>8s}")
print("  " + "-" * 32)

for name, svm in kernels.items():
    pipe = Pipeline([("s", StandardScaler()), ("svm", svm)])
    start = time.time()
    pipe.fit(X_d_train, y_d_train)
    elapsed = time.time() - start
    acc = accuracy_score(y_d_test, pipe.predict(X_d_test))
    print(f"  {name:<12s} {acc:>10.4f} {elapsed:>7.3f}s")

# Best model detailed report
best = Pipeline([("s", StandardScaler()), ("svm", SVC(kernel="rbf", C=10.0, gamma="scale"))])
best.fit(X_d_train, y_d_train)
y_pred = best.predict(X_d_test)

print(f"\n  Confusion Matrix (RBF kernel):")
cm = confusion_matrix(y_d_test, y_pred)
print(f"  Overall Accuracy: {accuracy_score(y_d_test, y_pred):.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. SVC uses OvO by default; LinearSVC uses OvR
  2. OvO: more classifiers but trained on smaller subsets
  3. OvR: fewer classifiers but trained on full dataset
  4. probability=True enables predict_proba() (via Platt scaling)
  5. For digits/MNIST: RBF kernel works very well
  6. LinearSVC is preferred for large multi-class problems

📚 Next: 04_svm_regression.py (SVR)
""")
