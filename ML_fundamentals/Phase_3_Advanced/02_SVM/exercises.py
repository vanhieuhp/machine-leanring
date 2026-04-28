"""
SVM - Exercises
================

Practice problems to reinforce your understanding of SVM.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, make_blobs, make_moons, load_iris, load_breast_cancer
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# ============================================================================
# EXERCISE 1: Iris Dataset with SVM
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Iris Dataset with SVM")
print("=" * 70)

print("""
Task:
1. Load the iris dataset (use first 2 features)
2. Convert to binary classification (setosa vs others)
3. Split into train/test (80/20)
4. Scale the features
5. Train SVM with linear kernel
6. Evaluate accuracy
""")

# Your code here:
iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, svm.predict(X_test_scaled))
print(f"\nAccuracy: {accuracy:.4f}")

# ============================================================================
# EXERCISE 2: Compare Linear vs RBF Kernel
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Compare Linear vs RBF Kernel")
print("=" * 70)

print("""
Task:
1. Generate moon-shaped data (make_moons)
2. Train SVM with linear kernel
3. Train SVM with RBF kernel
4. Compare accuracies
5. Visualize decision boundaries
""")

# Your code here:
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear kernel
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
linear_acc = accuracy_score(y_test, svm_linear.predict(X_test_scaled))

# RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0)
svm_rbf.fit(X_train_scaled, y_train)
rbf_acc = accuracy_score(y_test, svm_rbf.predict(X_test_scaled))

print(f"\nLinear Kernel Accuracy: {linear_acc:.4f}")
print(f"RBF Kernel Accuracy: {rbf_acc:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, svm, title in zip(axes, [svm_linear, svm_rbf], ['Linear', 'RBF']):
    ax.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm')
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='green')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                          np.linspace(ylim[0], ylim[1], 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    ax.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])
    ax.set_title(title)

plt.tight_layout()
plt.show()

# ============================================================================
# EXERCISE 3: Tune C and Gamma Parameters
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Tune C and Gamma Parameters")
print("=" * 70)

print("""
Task:
1. Use breast cancer dataset
2. Perform grid search for:
   - C: [0.1, 1, 10]
   - gamma: ['scale', 'auto', 0.01]
3. Use 5-fold cross-validation
4. Report best parameters
""")

# Your code here:
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01]
}

svm = SVC(kernel='rbf')
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.best_estimator_.score(X_test_scaled, y_test):.4f}")

# ============================================================================
# EXERCISE 4: SVM Regression
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: SVM Regression")
print("=" * 70)

print("""
Task:
1. Generate simple 1D regression data (y = sin(x) + noise)
2. Split into train/test
3. Train SVR with RBF kernel
4. Plot the regression line with data points
""")

# Your code here:
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr = SVR(kernel='rbf', C=10, epsilon=0.1)
svr.fit(X_train, y_train)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test')
plt.plot(X_train, svr.predict(X_train), color='red', linewidth=2, label='SVR')
plt.legend()
plt.title('SVR Regression')
plt.show()

from sklearn.metrics import r2_score
print(f"\nR² Score: {r2_score(y_test, svr.predict(X_test)):.4f}")

# ============================================================================
# EXERCISE 5: Visualize Decision Boundaries
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Visualize Decision Boundaries")
print("=" * 70)

print("""
Task:
1. Generate 2D blob data with 3 classes
2. Train SVM with RBF kernel
3. Create a contour plot showing decision boundaries
4. Show support vectors
""")

# Your code here:
X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_scaled, y_train)

# Plot decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='red', linewidths=2, label='Support Vectors')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                      np.linspace(ylim[0], ylim[1], 200))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
plt.title('SVM Decision Boundaries (3 Classes)')
plt.legend()
plt.show()

print(f"\nAccuracy: {accuracy_score(y_test, svm.predict(scaler.transform(X_test))):.4f}")

# ============================================================================
# EXERCISE 6: Compare Different Kernels
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Compare Different Kernels")
print("=" * 70)

print("""
Task:
1. Create three different datasets:
   - Linearly separable
   - Moons
   - Circles
2. Test all kernels on each dataset
3. Create a comparison table
""")

# Your code here:
from sklearn.datasets import make_circles

datasets = [
    ('Linear', make_classification(n_samples=200, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1, random_state=42)),
    ('Moons', make_moons(n_samples=200, noise=0.1, random_state=42)),
    ('Circles', make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42))
]

kernels = ['linear', 'rbf', 'poly']

print("\nComparison Table:")
print("-" * 50)

for name, (X, y) in datasets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   Scaler()
    X_train_scaled = scal scaler = Standarder.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n{name}:")
    for kernel in kernels:
        svm = SVC(kernel=kernel)
        svm.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test_scaled))
        print(f"  {kernel}: {acc:.4f}")

# ============================================================================
# BONUS: Multi-class Classification
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: Multi-class Classification")
print("=" * 70)

print("""
Task:
1. Use iris dataset (all 3 classes)
2. Train SVM with RBF kernel
3. Use One-vs-Rest strategy
4. Evaluate with classification report
""")

# Your code here:
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_scaled, y_train)

y_pred = svm.predict(X_test_scaled)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
What you practiced:
1. SVM classification on iris dataset
2. Comparing linear vs RBF kernel
3. Hyperparameter tuning (C, gamma)
4. SVM regression (SVR)
5. Decision boundary visualization
6. Multi-class classification
7. Different kernels on different data types

Key Takeaways:
1. Always scale features for SVM
2. RBF is good default for non-linear data
3. Linear works for high-dimensional, separable data
4. C and gamma control complexity
5. SVM can handle multi-class (OvR by default)
""")
