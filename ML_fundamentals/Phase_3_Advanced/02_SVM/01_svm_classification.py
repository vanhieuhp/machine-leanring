"""
SVM Classification - Deep Dive
===============================

This module covers:
- Linear SVM for classification
- Understanding margins and support vectors
- Hyperparameter C (regularization)
- Decision boundary visualization
- Practical examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. SVM CONCEPT
# ============================================================================

print("=" * 70)
print("1. SVM CONCEPT")
print("=" * 70)

print("""
Support Vector Machine (SVM):
- Finds the optimal hyperplane that separates classes
- Maximizes the margin between classes
- Works well in high-dimensional spaces

Key Concepts:
1. Hyperplane: Decision boundary (line in 2D, plane in 3D)
2. Margin: Distance from hyperplane to nearest points
3. Support Vectors: Points that define the margin
4. Optimal Hyperplane: Hyperplane with maximum margin

The Goal:
- Separate classes with the largest possible margin
- Support vectors are the critical data points
""")

# ============================================================================
# 2. LINEAR SVM
# ============================================================================

print("\n" + "=" * 70)
print("2. LINEAR SVM")
print("=" * 70)

# Generate linearly separable data
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Visualize data
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)
plt.title('Linearly Separable Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Train Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X, y)

# Get support vectors
support_vectors = svm_linear.support_vectors_
print(f"\nNumber of support vectors: {len(support_vectors)}")
print(f"Support vectors:\n{support_vectors}")

# Visualize decision boundary
def plot_svm_boundary(X, y, svm_model, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, alpha=0.7)

    # Plot support vectors
    plt.scatter(svm_model.support_vectors_[:, 0],
                svm_model.support_vectors_[:, 1],
                s=200, facecolors='none', edgecolors='green', linewidths=2,
                label='Support Vectors')

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create mesh grid
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
                linestyles=['--', '-', '--'])
    plt.title(title)
    plt.legend()
    plt.show()

plot_svm_boundary(X, y, svm_linear, 'Linear SVM')

# ============================================================================
# 3. C PARAMETER (REGULARIZATION)
# ============================================================================

print("\n" + "=" * 70)
print("3. C PARAMETER (REGULARIZATION)")
print("=" * 70)

print("""
C Parameter:
- Controls the trade-off between:
  1. Maximizing the margin
  2. Minimizing classification errors
- Low C: Large margin, more misclassifications allowed
- High C: Small margin, fewer misclassifications allowed

Analogy:
- Low C: "I don't want to be too strict, allow some errors"
- High C: "I want perfect classification, even if margin is small"

When to use:
- Low C: Noisy data, prevent overfitting
- High C: Clean data, want precise separation
""")

# Generate non-perfectly separable data
X_np, y_np = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=2.5)

# Compare different C values
C_values = [0.01, 0.1, 1, 10, 100]

plt.figure(figsize=(15, 5))
for i, C in enumerate(C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_np, y_np)

    plt.subplot(1, 5, i + 1)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm', s=30)
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='green')

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])
    plt.title(f'C={C}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# 4. PRACTICAL EXAMPLE: IRIS DATASET
# ============================================================================

print("\n" + "=" * 70)
print("4. PRACTICAL EXAMPLE: IRIS DATASET")
print("=" * 70)

# Load iris dataset
iris = load_iris()
X_iris = iris.data[:, :2]  # Use first 2 features
y_iris = (iris.target != 0).astype(int)  # Binary: setosa vs others

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm_iris = SVC(kernel='linear', C=1.0)
svm_iris.fit(X_train_scaled, y_train)

# Evaluate
y_pred = svm_iris.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize
plt.figure(figsize=(10, 5))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', s=50, alpha=0.7)
plt.scatter(svm_iris.support_vectors_[:, 0], svm_iris.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='green', label='Support Vectors')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = svm_iris.decision_function(np.c_[xx.ravel(), yy.ravel()])
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
plt.title('SVM on Iris Dataset')
plt.legend()
plt.show()

# ============================================================================
# 5. MULTI-CLASS SVM
# ============================================================================

print("\n" + "=" * 70)
print("5. MULTI-CLASS SVM")
print("=" * 70)

print("""
SVM is inherently binary. For multi-class:
1. One-vs-Rest (OvR): Train one classifier per class
2. One-vs-One (OvO): Train one classifier per pair of classes

sklearn uses One-vs-Rest by default.
""")

# Use full iris dataset (3 classes)
X_iris_full = iris.data[:, :2]
y_iris_full = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_iris_full, y_iris_full, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multi-class SVM
svm_multi = SVC(kernel='linear', C=1.0)
svm_multi.fit(X_train_scaled, y_train)

print(f"Multi-class Accuracy: {svm_multi.score(X_test_scaled, y_test):.4f}")
print(f"Number of support vectors per class: {svm_multi.n_support_}")

# ============================================================================
# 6. SVM WITH DIFFERENT C VALUES
# ============================================================================

print("\n" + "=" * 70)
print("6. TUNING C PARAMETER")
print("=" * 70)

# Compare C values with cross-validation
print("\nCross-validation scores for different C values:")
print("-" * 40)

for C in [0.1, 1, 10]:
    svm = SVC(kernel='linear', C=C)
    scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    print(f"C={C:5}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# ============================================================================
# 7. ADVANCED: MARGIN WIDTH
# ============================================================================

print("\n" + "=" * 70)
print("7. MARGIN VISUALIZATION")
print("=" * 70)

print("""
Margin Width:
- The margin is 2 / ||w|| where w is the weight vector
- Larger margin = better generalization
- C controls the margin width indirectly

Visualization:
- Solid line: Decision boundary
- Dashed lines: Margin boundaries
- Points on dashed lines: Support vectors
""")

# Visualize margins
X_m, y_m = make_blobs(n_samples=80, centers=2, random_state=42, cluster_std=1.2)
scaler = StandardScaler()
X_m_scaled = scaler.fit_transform(X_m)

svm_m = SVC(kernel='linear', C=1.0)
svm_m.fit(X_m_scaled, y_m)

plt.figure(figsize=(10, 5))
plt.scatter(X_m_scaled[:, 0], X_m_scaled[:, 1], c=y_m, cmap='coolwarm', s=50)

# Support vectors
plt.scatter(svm_m.support_vectors_[:, 0], svm_m.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='green', linewidths=2,
            label='Support Vectors')

# Decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                      np.linspace(ylim[0], ylim[1], 100))
Z = svm_m.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
plt.title('SVM Margin Visualization')
plt.legend()
plt.show()

# ============================================================================
# 8. CLASSIFICATION REPORT
# ============================================================================

print("\n" + "=" * 70)
print("8. DETAILED EVALUATION")
print("=" * 70)

# More detailed evaluation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Use breast cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_cancer = SVC(kernel='linear', C=1.0)
svm_cancer.fit(X_train_scaled, y_train)
y_pred = svm_cancer.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ============================================================================
# 9. COMPARISON: LINEAR VS OTHER KERNELS
# ============================================================================

print("\n" + "=" * 70)
print("9. LINEAR SVM PERFORMANCE")
print("=" * 70)

# Generate more complex data
X_complex, y_complex = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_complex, y_complex, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
linear_acc = svm_linear.score(X_test_scaled, y_test)

print(f"\nLinear SVM Accuracy: {linear_acc:.4f}")
print(f"Support vectors: {len(svm_linear.support_vectors_)}")

print("\n" + "=" * 70)
print("SVM CLASSIFICATION SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. SVM finds optimal hyperplane with maximum margin
2. Support vectors define the decision boundary
3. C parameter controls margin width vs misclassification
4. Always scale features for SVM
5. Linear SVM works well for linearly separable data
6. Use cross-validation to tune C

When to use Linear SVM:
- When data is linearly separable
- When you need interpretability
- For high-dimensional data (text classification)
- As baseline before trying non-linear kernels

Common Pitfalls:
- Forgetting to scale features
- Using wrong kernel
- Not tuning C
- Using too large dataset (SVM is slow)
""")
