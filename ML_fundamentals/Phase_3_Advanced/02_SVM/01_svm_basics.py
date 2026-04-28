"""
=================================================================
01 - SVM BASICS: Linear SVM, Margins, and the C Parameter
=================================================================
Topics:
  1. Linear SVM fundamentals
  2. Hard margin vs soft margin
  3. The C parameter effect
  4. Feature scaling importance
  5. Support vectors analysis
  6. Decision function and margins
=================================================================
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Section 1: Linear SVM Fundamentals ───────────────────────────
print("=" * 65)
print("SECTION 1: Linear SVM Fundamentals")
print("=" * 65)

# Create a simple linearly separable dataset
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Basic Linear SVM
svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)

print(f"\n  Training Accuracy: {accuracy_score(y_train, svm_linear.predict(X_train)):.4f}")
print(f"  Test Accuracy:     {accuracy_score(y_test, svm_linear.predict(X_test)):.4f}")
print(f"\n  Number of support vectors: {svm_linear.n_support_}")
print(f"  Total training samples:    {len(X_train)}")
print(f"  % support vectors:         {sum(svm_linear.n_support_) / len(X_train) * 100:.1f}%")

# ── Section 2: The C Parameter ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: The C Parameter (Regularization)")
print("=" * 65)

print("""
  C controls the trade-off between:
    • Maximizing margin width (C small → wide margin, some errors OK)
    • Minimizing misclassification (C large → narrow margin, fewer errors)
""")

# Create noisier dataset to see C's effect
X_noisy, y_noisy = make_classification(
    n_samples=300, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1,
    flip_y=0.15,  # 15% label noise
    random_state=42
)
X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(
    X_noisy, y_noisy, test_size=0.2, random_state=42
)

print(f"  {'C':>8s} {'Train Acc':>10s} {'Test Acc':>10s} {'# SVs':>8s} {'Interpretation':>20s}")
print("  " + "-" * 60)

for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    svm = SVC(kernel="linear", C=C, random_state=42)
    svm.fit(X_n_train, y_n_train)
    train_acc = accuracy_score(y_n_train, svm.predict(X_n_train))
    test_acc = accuracy_score(y_n_test, svm.predict(X_n_test))
    n_sv = sum(svm.n_support_)

    if C <= 0.01:
        interp = "Underfitting"
    elif C >= 100:
        interp = "Overfitting risk"
    else:
        interp = "Good balance"

    print(f"  {C:>8.3f} {train_acc:>10.4f} {test_acc:>10.4f} {n_sv:>8d} {interp:>20s}")

print("\n  💡 Notice: More SVs with small C (wide margin)")
print("     Fewer SVs with large C (narrow margin, tight fit)")

# ── Section 3: Feature Scaling Importance ─────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Feature Scaling — CRITICAL for SVM!")
print("=" * 65)

# Use breast cancer dataset
data = load_breast_cancer()
X_bc, y_bc = data.data, data.target
X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42
)

# WITHOUT scaling
svm_noscale = SVC(kernel="rbf", C=1.0, random_state=42)
svm_noscale.fit(X_bc_train, y_bc_train)
acc_noscale = accuracy_score(y_bc_test, svm_noscale.predict(X_bc_test))

# WITH scaling
scaler = StandardScaler()
X_bc_train_scaled = scaler.fit_transform(X_bc_train)
X_bc_test_scaled = scaler.transform(X_bc_test)
svm_scaled = SVC(kernel="rbf", C=1.0, random_state=42)
svm_scaled.fit(X_bc_train_scaled, y_bc_train)
acc_scaled = accuracy_score(y_bc_test, svm_scaled.predict(X_bc_test_scaled))

# WITH Pipeline (recommended)
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, random_state=42))
])
svm_pipeline.fit(X_bc_train, y_bc_train)
acc_pipeline = accuracy_score(y_bc_test, svm_pipeline.predict(X_bc_test))

print(f"\n  Without scaling:  {acc_noscale:.4f}")
print(f"  With scaling:     {acc_scaled:.4f}")
print(f"  With Pipeline:    {acc_pipeline:.4f}")
print(f"\n  ⚠️  Improvement from scaling: {(acc_scaled - acc_noscale) * 100:.1f}%!")

# Show feature ranges to understand why
print("\n  Feature value ranges (first 5 features):")
for i in range(5):
    print(f"    {data.feature_names[i]:>25s}: "
          f"[{X_bc_train[:, i].min():>8.2f}, {X_bc_train[:, i].max():>8.2f}]")
print("  → Features are on very different scales!")

# ── Section 4: Support Vectors Analysis ───────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Understanding Support Vectors")
print("=" * 65)

X_simple, y_simple = make_classification(
    n_samples=100, n_features=2, n_informative=2,
    n_redundant=0, random_state=42
)
svm = SVC(kernel="linear", C=1.0, random_state=42)
svm.fit(X_simple, y_simple)

print(f"\n  Total training samples:    {len(X_simple)}")
print(f"  Number of support vectors: {sum(svm.n_support_)}")
print(f"  SVs per class:             {svm.n_support_}")
print(f"  Support vector indices:    {svm.support_[:10]}...")

# Show distances to decision boundary
distances = svm.decision_function(X_simple)
print(f"\n  Decision function values (first 10): {np.round(distances[:10], 3)}")
print("  Positive → Class 1 side, Negative → Class 0 side")
print("  Values close to 0 → near the boundary")

# Support vectors are closest to boundary
sv_distances = np.abs(svm.decision_function(svm.support_vectors_))
print(f"\n  SV distances to boundary: min={sv_distances.min():.3f}, max={sv_distances.max():.3f}")
print("  💡 Support vectors are the closest points to the boundary")

# ── Section 5: SVC vs LinearSVC ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: SVC vs LinearSVC")
print("=" * 65)

print("""
  ┌────────────────┬──────────────────────┬──────────────────────┐
  │                │ SVC(kernel='linear') │ LinearSVC            │
  ├────────────────┼──────────────────────┼──────────────────────┤
  │ Implementation │ libsvm               │ liblinear            │
  │ Speed          │ Slower               │ Faster               │
  │ Scalability    │ O(n²-n³)             │ O(n)                 │
  │ Multi-class    │ One-vs-One           │ One-vs-Rest          │
  │ Probability    │ Yes (probability=T)  │ No                   │
  │ Kernel trick   │ Yes                  │ No (linear only)     │
  │ Best for       │ Small datasets       │ Large datasets       │
  └────────────────┴──────────────────────┴──────────────────────┘
""")

# Speed comparison
import time

X_big, y_big = make_classification(n_samples=5000, n_features=20, random_state=42)
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_b_train_s = scaler.fit_transform(X_b_train)
X_b_test_s = scaler.transform(X_b_test)

start = time.time()
SVC(kernel="linear", C=1.0).fit(X_b_train_s, y_b_train)
svc_time = time.time() - start

start = time.time()
LinearSVC(C=1.0, max_iter=5000).fit(X_b_train_s, y_b_train)
lsvc_time = time.time() - start

print(f"  SVC(linear) time:   {svc_time:.3f}s")
print(f"  LinearSVC time:     {lsvc_time:.3f}s")
print(f"  Speedup:            {svc_time / lsvc_time:.1f}x")

# ── Section 6: Practical Example ──────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Complete Practical Example")
print("=" * 65)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build proper pipeline
best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42))
])

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

print(f"\n  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  CV Score:      {cross_val_score(best_pipeline, X_train, y_train, cv=5).mean():.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names, indent=4))

# ── Summary ───────────────────────────────────────────────────────
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. ALWAYS scale features before SVM (use Pipeline!)
  2. C controls regularization: small C → wide margin → simpler
  3. Support vectors are the only points that matter
  4. LinearSVC is much faster for large datasets
  5. SVC with RBF kernel is a good default choice

📚 Next: 02_kernels.py (Kernel Trick Deep Dive)
""")
