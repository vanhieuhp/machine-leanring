"""
Decision Trees - Part 1: Classification Trees
==============================================

This module covers:
- Decision tree basics
- Tree building algorithm
- Splitting criteria
- Tree visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================================
# 1. DECISION TREE BASICS
# ============================================================================

print("=" * 70)
print("1. DECISION TREE BASICS")
print("=" * 70)

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Feature names: {iris.feature_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================================
# 2. TRAIN DECISION TREE
# ============================================================================

print("\n" + "=" * 70)
print("2. TRAIN DECISION TREE")
print("=" * 70)

# Train with default parameters
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predictions
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")
print(f"Accuracy: {accuracy:.4f}")

# ============================================================================
# 3. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("3. FEATURE IMPORTANCE")
print("=" * 70)

importances = tree.feature_importances_
feature_names = iris.feature_names

print("Feature Importance:")
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# ============================================================================
# 4. CONTROLLING TREE DEPTH
# ============================================================================

print("\n" + "=" * 70)
print("4. CONTROLLING TREE DEPTH")
print("=" * 70)

depths = [1, 3, 5, 10, None]
train_accuracies = []
test_accuracies = []

for depth in depths:
    tree_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_depth.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, tree_depth.predict(X_train))
    test_acc = accuracy_score(y_test, tree_depth.predict(X_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    depth_str = str(depth) if depth else "None"
    print(f"Max depth {depth_str}: Train={train_acc:.4f}, Test={test_acc:.4f}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("5. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Tree structure (simplified)
ax = axes[0, 0]
tree_simple = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_simple.fit(X_train, y_train)
plot_tree(tree_simple, feature_names=iris.feature_names, class_names=iris.target_names,
          ax=ax, filled=True, fontsize=10)
ax.set_title('Decision Tree (max_depth=3)')

# Plot 2: Feature importance
ax = axes[0, 1]
ax.barh(feature_names, importances, color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance')
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Train vs Test accuracy
ax = axes[1, 0]
depth_labels = [str(d) if d else "None" for d in depths]
ax.plot(range(len(depths)), train_accuracies, marker='o', label='Train', linewidth=2)
ax.plot(range(len(depths)), test_accuracies, marker='s', label='Test', linewidth=2)
ax.set_xticks(range(len(depths)))
ax.set_xticklabels(depth_labels)
ax.set_xlabel('Max Depth')
ax.set_ylabel('Accuracy')
ax.set_title('Train vs Test Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Confusion matrix
ax = axes[1, 1]
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(iris.target_names)
ax.set_yticklabels(iris.target_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=14)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# ============================================================================
# 6. CLASSIFICATION REPORT
# ============================================================================

print("\n" + "=" * 70)
print("6. CLASSIFICATION REPORT")
print("=" * 70)

print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ============================================================================
# 7. PRACTICAL EXAMPLE: CUSTOMER SEGMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL EXAMPLE: CUSTOMER SEGMENTATION")
print("=" * 70)

# Create customer data
np.random.seed(42)
n_customers = 200

# High value customers
age_high = np.random.normal(45, 10, n_customers // 2)
income_high = np.random.normal(100000, 20000, n_customers // 2)
y_high = np.ones(n_customers // 2)

# Low value customers
age_low = np.random.normal(30, 10, n_customers // 2)
income_low = np.random.normal(40000, 15000, n_customers // 2)
y_low = np.zeros(n_customers // 2)

X_customers = np.vstack([
    np.column_stack([age_high, income_high]),
    np.column_stack([age_low, income_low])
])
y_customers = np.hstack([y_high, y_low])

# Train tree
X_train_cust, X_test_cust, y_train_cust, y_test_cust = train_test_split(
    X_customers, y_customers, test_size=0.2, random_state=42
)

tree_cust = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_cust.fit(X_train_cust, y_train_cust)

accuracy_cust = accuracy_score(y_test_cust, tree_cust.predict(X_test_cust))
print(f"Customer segmentation accuracy: {accuracy_cust:.4f}")

# Visualize decision boundary
plt.figure(figsize=(10, 8))

# Plot data
plt.scatter(X_train_cust[y_train_cust == 0, 0], X_train_cust[y_train_cust == 0, 1],
           alpha=0.6, label='Low value', color='blue')
plt.scatter(X_train_cust[y_train_cust == 1, 0], X_train_cust[y_train_cust == 1, 1],
           alpha=0.6, label='High value', color='red')

# Decision boundary
x_min, x_max = X_train_cust[:, 0].min() - 5, X_train_cust[:, 0].max() + 5
y_min, y_max = X_train_cust[:, 1].min() - 5000, X_train_cust[:, 1].max() + 5000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = tree_cust.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.title('Customer Segmentation Decision Tree')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
