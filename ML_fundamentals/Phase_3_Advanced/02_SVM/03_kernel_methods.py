"""
Kernel Methods - Deep Dive
===========================

This module covers:
- The Kernel Trick
- Linear, RBF, Polynomial kernels
- Choosing the right kernel
- Visualizing kernel transformations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. THE KERNEL TRICK
# ============================================================================

print("=" * 70)
print("1. THE KERNEL TRICK")
print("=" * 70)

print("""
The Kernel Trick:
- Instead of transforming data to high dimension explicitly
- Compute dot products in high dimension directly
- Much more efficient!

Why it works:
- Many algorithms only need dot products
- Kernel function K(x, y) = φ(x) · φ(y)
- Never explicitly compute φ(x)!

Benefits:
- Handle non-linear relationships
- Computational efficiency
- Memory efficiency
""")

# ============================================================================
# 2. LINEAR VS NON-LINEAR DATA
# ============================================================================

print("\n" + "=" * 70)
print("2. LINEAR VS NON-LINEAR DATA")
print("=" * 70)

# Generate different types of data
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Linear separable
X_linear, y_linear = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                         n_informative=2, n_clusters_per_class=1,
                                         random_state=42)
axes[0].scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, cmap='coolwarm')
axes[0].set_title('Linearly Separable')

# Moons
X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='coolwarm')
axes[1].set_title('Moons (Non-linear)')

# Circles
X_circles, y_circles = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
axes[2].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='coolwarm')
axes[2].set_title('Circles (Non-linear)')

plt.tight_layout()
plt.show()

# ============================================================================
# 3. LINEAR KERNEL
# ============================================================================

print("\n" + "=" * 70)
print("3. LINEAR KERNEL")
print("=" * 70)

print("""
Linear Kernel:
K(x, y) = x · y

- Works like standard linear classifier
- Best for linearly separable data
- Fastest kernel (no transformation)
- Good for high-dimensional data (text, genomics)
""")

# Test on linearly separable data
X_train, X_test, y_train, y_test = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear kernel
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
linear_acc = accuracy_score(y_test, svm_linear.predict(X_test_scaled))

print(f"Linear Kernel Accuracy: {linear_acc:.4f}")

# Visualize
plt.figure(figsize=(10, 5))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
plt.scatter(svm_linear.support_vectors_[:, 0], svm_linear.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='green', label='Support Vectors')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = svm_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])
plt.title('Linear Kernel')
plt.legend()
plt.show()

# ============================================================================
# 4. RBF KERNEL
# ============================================================================

print("\n" + "=" * 70)
print("4. RBF (RADIAL BASIS FUNCTION) KERNEL")
print("=" * 70)

print("""
RBF Kernel:
K(x, y) = exp(-γ ||x - y||²)

- Also called Gaussian kernel
- Most popular kernel
- Infinite dimensional transformation (theoretically)
- γ (gamma): controls influence of each training example

Effect of gamma:
- High γ: narrow bell curve, complex decision boundary
- Low γ: wide bell curve, smoother decision boundary
""")

# Test on moons data
X_train, X_test, y_train, y_test = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
rbf_acc = accuracy_score(y_test, svm_rbf.predict(X_test_scaled))

print(f"RBF Kernel Accuracy (Moons): {rbf_acc:.4f}")

# Compare gamma values
gammas = [0.1, 1, 10, 'scale', 'auto']

plt.figure(figsize=(15, 5))

for i, gamma in enumerate(gammas):
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma)
    svm.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test_scaled))

    plt.subplot(1, 5, i + 1)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', alpha=0.5)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])
    plt.title(f'γ = {gamma}\nAcc = {acc:.2f}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# 5. POLYNOMIAL KERNEL
# ============================================================================

print("\n" + "=" * 70)
print("5. POLYNOMIAL KERNEL")
print("=" * 70)

print("""
Polynomial Kernel:
K(x, y) = (x · y + c)^d

- c: constant term (default 1)
- d: degree of polynomial
- Creates polynomial decision boundaries

Common settings:
- degree=2: quadratic
- degree=3: cubic
- c=1: homogeneous
""")

# Test polynomial kernel
degrees = [2, 3, 4]

plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    svm = SVC(kernel='poly', C=1.0, degree=degree, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test_scaled))

    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', alpha=0.5)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])
    plt.title(f'degree = {degree}\nAcc = {acc:.2f}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# 6. CIRCLES DATA
# ============================================================================

print("\n" + "=" * 70)
print("6. HANDLING CIRCULAR DATA")
print("=" * 70)

print("""
Circles data:
- One class in center, one class in ring
- Not linearly separable
- RBF kernel works well!
""")

# Test on circles
X_train, X_test, y_train, y_test = train_test_split(
    X_circles, y_circles, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear - will fail
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
linear_acc = accuracy_score(y_test, svm_linear.predict(X_test_scaled))

# RBF - will work
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
rbf_acc = accuracy_score(y_test, svm_rbf.predict(X_test_scaled))

print(f"Linear Kernel: {linear_acc:.4f}")
print(f"RBF Kernel: {rbf_acc:.4f}")

# Visualize RBF on circles
plt.figure(figsize=(10, 5))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
plt.title('RBF Kernel on Circular Data')
plt.show()

# ============================================================================
# 7. KERNEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("7. KERNEL COMPARISON")
print("=" * 70)

print("""
When to use which kernel:

1. Linear:
   - Linearly separable data
   - High-dimensional data (text, genomics)
   - Large datasets
   - When interpretability matters

2. RBF (default):
   - Non-linear data
   - Most cases (good default)
   - Many features
   - When you don't know the pattern

3. Polynomial:
   - When you suspect polynomial relationship
   - Customizable with degree
   - Less commonly used
""")

# ============================================================================
# 8. SIGMOID KERNEL
# ============================================================================

print("\n" + "=" * 70)
print("8. SIGMOID KERNEL")
print("=" * 70)

print("""
Sigmoid Kernel:
K(x, y) = tanh(α · x · y + c)

- Similar to neural network activation
- Not always positive definite
- Less commonly used
""")

# ============================================================================
# 9. PRACTICAL GUIDELINES
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL GUIDELINES")
print("=" * 70)

print("""
Step-by-step approach:

1. Start with RBF kernel (default choice)
2. Scale your data! (critical for SVM)
3. Use grid search for:
   - C: [0.1, 1, 10, 100]
   - gamma: ['scale', 'auto', 0.01, 0.1, 1]
4. Consider:
   - Linear if data is high-dimensional
   - RBF for most cases
   - Polynomial for specific patterns

Remember:
- Always scale features
- RBF is a safe default
- Gamma and C are related (large C = small gamma)
""")

# ============================================================================
# 10. COMPLETE COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("10. COMPLETE COMPARISON")
print("=" * 70)

# Generate all three types
datasets = [
    ('Linear', make_classification(n_samples=200, n_features=2, n_redundant=0,
                                    n_informative=2, n_clusters_per_class=1, random_state=42)),
    ('Moons', make_moons(n_samples=200, noise=0.1, random_state=42)),
    ('Circles', make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42))
]

kernels = ['linear', 'rbf', 'poly']

print("\nDataset vs Kernel Accuracy:")
print("-" * 50)

for name, (X, y) in datasets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{name} Data:")
    for kernel in kernels:
        svm = SVC(kernel=kernel, C=1.0, gamma='scale')
        try:
            svm.fit(X_train_scaled, y_train)
            acc = accuracy_score(y_test, svm.predict(X_test_scaled))
            print(f"  {kernel}: {acc:.4f}")
        except:
            print(f"  {kernel}: failed")

print("\n" + "=" * 70)
print("KERNEL METHODS SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. Kernel trick: compute in high dimensions efficiently
2. RBF is default, works for most non-linear cases
3. Linear for high-dimensional, separable data
4. Polynomial for specific polynomial patterns
5. Always scale data!
6. Grid search for C and gamma

Common Pitfalls:
- Forgetting to scale features
- Using wrong kernel
- Not tuning C and gamma
- Overfitting with small datasets
""")
