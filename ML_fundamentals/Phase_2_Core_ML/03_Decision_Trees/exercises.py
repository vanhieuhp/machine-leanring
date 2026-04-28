"""
Decision Trees - Exercises
==========================

Practice problems for Decision Trees.
Solutions are provided at the bottom.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXERCISE 1: Classification Tree Basics
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Classification Tree Basics")
print("=" * 70)

# 1.1 Create simple dataset: weather -> play tennis
# Features: Outlook (sunny/rainy), Temperature (hot/mild)

data = {
    'outlook': ['sunny', 'sunny', 'rainy', 'rainy', 'sunny', 'rainy', 'sunny', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'play': [0, 0, 1, 1, 1, 1, 0, 1]  # 0=no, 1=yes
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)

# 1.2 Convert categorical to numerical
# Hint: Use pd.Categorical or manual mapping
# TODO: Convert

# 1.3 Train decision tree classifier
# TODO: Train

# 1.4 Predict for: sunny + mild weather
# TODO: Predict

# 1.5 Visualize tree (optional - just print rules)
# Hint: Use model.tree_.feature to get split features

# ============================================================================
# EXERCISE 2: Controlling Tree Depth
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Controlling Tree Depth")
print("=" * 70)

# 2.1 Create larger classification dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                          n_redundant=2, random_state=42)

# 2.2 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2.3 Train trees with different depths: 1, 3, 5, 10, None
depths = [1, 3, 5, 10, None]

print("Depth | Train Acc | Test Acc | Leaves")
print("-" * 45)

for depth in depths:
    # TODO: Train tree with max_depth=depth

    # TODO: Calculate train and test accuracy

    # TODO: Count leaves

    # Print results
    depth_str = str(depth) if depth else "None"
    # TODO: Print formatted results

# 2.4 Which depth gives best generalization?

# 2.5 What happens if depth is too high? (overfitting)

# ============================================================================
# EXERCISE 3: Feature Importance
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Feature Importance")
print("=" * 70)

# Using the same dataset from Exercise 2

# 3.1 Train a tree with depth=5
# TODO: Train

# 3.2 Get feature importances
# TODO: Get importances

# 3.3 Print feature importance for each feature
print("Feature Importance:")
for i, importance in enumerate(importances):
    print(f"  Feature {i}: {importance:.4f}")

# 3.4 Which features are most important?

# 3.5 Create horizontal bar chart of importance
# TODO: Plot

# ============================================================================
# EXERCISE 4: Regression Trees
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Regression Trees")
print("=" * 70)

# 4.1 Create regression dataset
X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

# 4.2 Split data
# TODO: Split

# 4.3 Train regression tree
# TODO: Train with max_depth=3

# 4.4 Predict on test set
# TODO: Predict

# 4.5 Calculate metrics: MSE, RMSE, R²
# TODO: Calculate

# 4.6 Compare with linear regression
# Hint: from sklearn.linear_model import LinearRegression

# TODO: Compare

# ============================================================================
# EXERCISE 5: Entropy and Information Gain
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Entropy and Information Gain")
print("=" * 70)

# 5.1 Implement entropy function
def entropy(y):
    """
    Calculate entropy.

    Args:
        y: Array of class labels

    Returns:
        Entropy value
    """
    # TODO: Implement
    # Hint: -sum(p * log2(p)) for each class
    pass

# Test
test_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 50% each class
print(f"Entropy for balanced: {entropy(test_labels):.4f}")

test_labels2 = np.array([0, 0, 0, 0, 0, 0, 0, 1])  # 87.5% class 0
print(f"Entropy for skewed: {entropy(test_labels2):.4f}")

# 5.2 Implement information gain
def information_gain(X, y, feature_idx, threshold):
    """
    Calculate information gain for a split.

    Args:
        X: Feature matrix
        y: Target labels
        feature_idx: Index of feature to split on
        threshold: Split threshold

    Returns:
        Information gain
    """
    # TODO: Implement
    # Hint: entropy(y) - weighted_avg(entropy(left) + entropy(right))
    pass

# ============================================================================
# EXERCISE 6: Handling Overfitting
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Handling Overfitting")
print("=" * 70)

# 6.1 Create dataset with noise
X_over, y_over = make_classification(n_samples=200, n_features=20, n_informative=5,
                                     n_redundant=10, n_clusters_per_class=2,
                                     random_state=42)

# 6.2 Split data
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

# 6.3 Train with different min_samples_leaf
min_samples = [1, 5, 10, 20, 50]

print("min_samples_leaf | Train Acc | Test Acc")
print("-" * 45)

for min_samp in min_samples:
    # TODO: Train with different min_samples_leaf

    # TODO: Calculate and print accuracies

    pass

# 6.4 What happens as min_samples_leaf increases?

# 6.5 Try max_features
print("\nmax_features | Train Acc | Test Acc")
print("-" * 45)

for max_feat in [1, 5, 10, 'sqrt', 'log2']:
    # TODO: Train with different max_features

    # TODO: Calculate and print accuracies

    pass

# ============================================================================
# EXERCISE 7: Real-world - Loan Approval
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Real-world - Loan Approval")
print("=" * 70)

# Simulated loan approval data
np.random.seed(42)
n = 500

loans = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n),
    'credit_score': np.random.normal(650, 100, n),
    'debt': np.random.normal(10000, 5000, n),
    'employment_years': np.random.exponential(5, n)
})

# Approval based on rules
loans['approved'] = (
    (loans['credit_score'] > 600) &
    (loans['debt'] / loans['income'] < 0.3) &
    (loans['employment_years'] > 1)
).astype(int)

print("Loan Dataset:")
print(loans.head())
print(f"\nApproval rate: {loans['approved'].mean():.2%}")

# 7.1 Split features and target
# TODO: X, y

# 7.2 Split train/test
# TODO: Split

# 7.3 Train decision tree
# TODO: Train

# 7.4 Evaluate
# TODO: Evaluate

# 7.5 Feature importance
# TODO: Print importance

# 7.6 Predict for new applicant:
# - Income: 60000, Credit: 700, Debt: 10000, Employment: 3 years
new_applicant = [[60000, 700, 10000, 3]]
# TODO: Predict

# ============================================================================
# EXERCISE 8: Visualizing Decision Trees
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Visualizing Decision Trees")
print("=" * 70)

# 8.1 Create simple dataset
X_vis, y_vis = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                   n_informative=2, random_state=42)

# 8.2 Train tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_vis, y_vis)

# 8.3 Plot using matplotlib
# Hint: Use plot_tree() from sklearn.tree

# TODO: Create plot

# 8.4 Plot decision boundary
# TODO: Create function to plot decision boundary

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("\n--- EXERCISE 1: Classification Tree Basics ---")
data = {
    'outlook': ['sunny', 'sunny', 'rainy', 'rainy', 'sunny', 'rainy', 'sunny', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'play': [0, 0, 1, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Convert to numerical
df['outlook_num'] = df['outlook'].map({'sunny': 0, 'rainy': 1})
df['temp_num'] = df['temperature'].map({'hot': 0, 'mild': 1})

X = df[['outlook_num', 'temp_num']]
y = df['play']

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)

# Predict for sunny + mild
prediction = tree.predict([[0, 1]])[0]
print(f"Prediction for sunny+mild: {'Play' if prediction == 1 else 'No Play'}")

print("\n--- EXERCISE 2: Controlling Tree Depth ---")
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                          n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

depths = [1, 3, 5, 10, None]
print("Depth | Train Acc | Test Acc | Leaves")
print("-" * 45)

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    n_leaves = tree.get_n_leaves()

    depth_str = str(depth) if depth else "None"
    print(f"{depth_str:4s} | {train_acc:.4f}   | {test_acc:.4f} | {n_leaves}")

print("\n--- EXERCISE 3: Feature Importance ---")
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

importances = tree.feature_importances_
print("Feature Importance:")
for i, imp in enumerate(importances):
    print(f"  Feature {i}: {imp:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- EXERCISE 4: Regression Trees ---")
X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression R²: {r2_lr:.4f}")

print("\n--- EXERCISE 5: Entropy and Information Gain ---")
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

test_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
print(f"Entropy for balanced: {entropy(test_labels):.4f}")

test_labels2 = np.array([0, 0, 0, 0, 0, 0, 0, 1])
print(f"Entropy for skewed: {entropy(test_labels2):.4f}")

print("\n--- EXERCISE 6: Handling Overfitting ---")
X_over, y_over = make_classification(n_samples=200, n_features=20, n_informative=5,
                                     n_redundant=10, n_clusters_per_class=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

print("min_samples_leaf | Train Acc | Test Acc")
print("-" * 45)

for min_samp in [1, 5, 10, 20, 50]:
    tree = DecisionTreeClassifier(min_samples_leaf=min_samp, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"{min_samp:17d} | {train_acc:.4f}   | {test_acc:.4f}")

print("\nmax_features | Train Acc | Test Acc")
print("-" * 45)

for max_feat in [1, 5, 10, 'sqrt', 'log2']:
    tree = DecisionTreeClassifier(max_features=max_feat, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"{str(max_feat):12s} | {train_acc:.4f}   | {test_acc:.4f}")

print("\n--- EXERCISE 7: Loan Approval ---")
np.random.seed(42)
n = 500

loans = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n),
    'credit_score': np.random.normal(650, 100, n),
    'debt': np.random.normal(10000, 5000, n),
    'employment_years': np.random.exponential(5, n)
})

loans['approved'] = (
    (loans['credit_score'] > 600) &
    (loans['debt'] / loans['income'] < 0.3) &
    (loans['employment_years'] > 1)
).astype(int)

X = loans[['income', 'credit_score', 'debt', 'employment_years']]
y = loans['approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, tree.predict(X_test)):.4f}")
print(f"Feature Importance: {dict(zip(X.columns, tree.feature_importances_))}")

new_applicant = [[60000, 700, 10000, 3]]
print(f"Prediction for new applicant: {tree.predict(new_applicant)[0]}")

print("\n--- EXERCISE 8: Visualizing Decision Trees ---")
X_vis, y_vis = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                   n_informative=2, random_state=42)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_vis, y_vis)

plt.figure(figsize=(15, 8))
plot_tree(tree, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()
