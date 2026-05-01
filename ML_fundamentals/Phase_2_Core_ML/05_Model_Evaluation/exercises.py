"""
Model Evaluation - Exercises
==========================

Practice problems for Model Evaluation.
Solutions are provided at the bottom.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc,
                           roc_auc_score, mean_squared_error, mean_absolute_error,
                           r2_score)
from sklearn.datasets import make_classification, make_regression

# ============================================================================
# EXERCISE 1: Train/Test Split
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Train/Test Split")
print("=" * 70)

# 1.1 Create classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 1.2 Split with different test sizes: 0.1, 0.2, 0.3, 0.4
test_sizes = [0.1, 0.2, 0.3, 0.4]

print("Test Size | Train Size | Test Size | Model Accuracy")
print("-" * 55)

for test_size in test_sizes:
    # TODO: Split data with different test sizes

    # TODO: Train model

    # TODO: Evaluate

    # Print results
    pass

# 1.3 How does test size affect reliability of evaluation?

# 1.4 What about different random states?
# TODO: Try same split with different random_state

# ============================================================================
# EXERCISE 2: Cross-Validation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Cross-Validation")
print("=" * 70)

# 2.1 Create dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# 2.2 5-fold cross-validation
# TODO: Use cross_val_score with cv=5

# 2.3 Calculate mean and std of scores
# TODO: Calculate

# 2.4 Compare with different cv values: 3, 5, 10
cv_values = [3, 5, 10]

print("\ncv | Mean Accuracy | Std")
print("-" * 30)

for cv in cv_values:
    # TODO: Cross-validation with different cv

    # TODO: Calculate mean and std

    # Print

    pass

# 2.5 Stratified K-Fold (important for imbalanced data)
# TODO: Use StratifiedKFold

# ============================================================================
# EXERCISE 3: Confusion Matrix Analysis
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Confusion Matrix Analysis")
print("=" * 70)

# 3.1 Create imbalanced dataset (95% class 0, 5% class 1)
X, y = make_classification(n_samples=500, weights=[0.95, 0.05], random_state=42)

# 3.2 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.3 Train model
# TODO: Train

# 3.4 Make predictions
# TODO: Predict

# 3.5 Calculate confusion matrix
# TODO: Calculate

# 3.6 Extract TP, TN, FP, FN
# TODO: Extract

# 3.7 Calculate metrics
# TODO: Calculate accuracy, precision, recall, f1

# 3.8 Why is accuracy misleading here?
# TODO: Analyze

# ============================================================================
# EXERCISE 4: ROC Curve and AUC
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: ROC Curve and AUC")
print("=" * 70)

# 4.1 Create dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4.2 Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4.3 Get probability predictions
# TODO: Get probabilities

# 4.4 Calculate ROC curve
# TODO: Calculate

# 4.5 Calculate AUC score
# TODO: Calculate

# 4.6 Plot ROC curve
# TODO: Plot

# 4.7 Compare with different thresholds
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

print("\nThreshold | Precision | Recall | F1")
print("-" * 45)

for threshold in thresholds:
    # TODO: Predict with threshold

    # TODO: Calculate metrics

    # Print

    pass

# ============================================================================
# EXERCISE 5: Regression Metrics
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Regression Metrics")
print("=" * 70)

# 5.1 Create regression dataset
X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 5.2 Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5.3 Make predictions
# TODO: Predict

# 5.4 Calculate metrics
# MSE
# TODO: Calculate

# RMSE
# TODO: Calculate

# MAE
# TODO: Calculate

# R² Score
# TODO: Calculate

# 5.5 Compare metrics - which is most interpretable?

# ============================================================================
# EXERCISE 6: Bias-Variance Tradeoff
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Bias-Variance Tradeoff")
print("=" * 70)

# 6.1 Create dataset
X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6.2 Train models with different depths
depths = [1, 2, 3, 5, 10, None]  # None = unlimited

print("Depth | Train MSE | Test MSE | Overfit?")
print("-" * 50)

for depth in depths:
    # TODO: Train DecisionTreeRegressor with different max_depth

    # TODO: Predict on train and test

    # TODO: Calculate MSE for train and test

    # Print results
    pass

# 6.3 Identify overfitting vs underfitting

# ============================================================================
# EXERCISE 7: Real-world - Medical Diagnosis
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Medical Diagnosis Evaluation")
print("=" * 70)

# Simulate medical diagnosis data
np.random.seed(42)
n_patients = 1000

# Features: test1, test2, test3, age
X = np.random.randn(n_patients, 4)

# Target: disease (1) or not (0)
# Disease probability based on features
disease_prob = 1 / (1 + np.exp(-(0.5*X[:, 0] + 0.3*X[:, 1] - 0.2*X[:, 2] + 0.01*X[:, 3])))
y = (np.random.rand(n_patients) < disease_prob).astype(int)

print(f"Disease prevalence: {y.mean():.2%}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
# TODO: Calculate accuracy

# TODO: Calculate precision

# TODO: Calculate recall

# TODO: Calculate F1

# TODO: Calculate ROC-AUC

# Print all metrics
print("\nMetrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Which metric is most important for medical diagnosis?
# - False negative (missed disease) is VERY BAD
# - False positive (false alarm) is less bad

# ============================================================================
# EXERCISE 8: Precision-Recall Tradeoff
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Precision-Recall Tradeoff")
print("=" * 70)

# Using medical diagnosis data
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
from sklearn.metrics import precision_recall_curve, average_precision_score

# TODO: Calculate precision-recall curve

# TODO: Calculate average precision

# Plot
# TODO: Plot

# ============================================================================
# EXERCISE 9: Leave-One-Out Cross-Validation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9: Leave-One-Out Cross-Validation")
print("=" * 70)

from sklearn.model_selection import LeaveOneOut

# Create small dataset
X_small = X[:50]
y_small = y[:50]

# TODO: Use LeaveOneOut

# TODO: Perform cross-validation

# Calculate mean accuracy
# TODO: Calculate

print(f"LOOCV Accuracy: {mean_accuracy:.4f}")

# Compare with 5-fold CV
# TODO: 5-fold CV

# Compare results

# ============================================================================
# EXERCISE 10: Model Comparison
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 10: Model Comparison")
print("=" * 70)

# 10.1 Create classification dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# 10.2 Define models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Naive Bayes': GaussianNB()
}

# 10.3 Evaluate each model with 5-fold CV
print("Model | Mean CV Accuracy | Std")
print("-" * 40)

for name, model in models.items():
    # TODO: Cross-validation

    # TODO: Calculate mean and std

    # Print
    pass

# 10.4 Which model is best?

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("\n--- EXERCISE 1: Train/Test Split ---")
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

test_sizes = [0.1, 0.2, 0.3, 0.4]

print("Test Size | Train Size | Test Size | Model Accuracy")
print("-" * 55)

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{test_size:.1f}      | {len(y_train):5d}      | {len(y_test):4d}    | {acc:.4f}")

print("\n--- EXERCISE 2: Cross-Validation ---")
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

model = LogisticRegression(max_iter=1000)

for cv in [3, 5, 10]:
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"cv={cv}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

print("\n--- EXERCISE 3: Confusion Matrix ---")
X, y = make_classification(n_samples=500, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix:\n{cm}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")
print("\nNote: High accuracy (94%) is misleading due to class imbalance!")

print("\n--- EXERCISE 4: ROC Curve ---")
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"ROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nThreshold Analysis:")
for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_pred = (y_proba >= threshold).astype(int)
    print(f"Threshold {threshold}: Precision={precision_score(y_test, y_pred):.4f}, Recall={recall_score(y_test, y_pred):.4f}, F1={f1_score(y_test, y_pred):.4f}")

print("\n--- EXERCISE 5: Regression Metrics ---")
X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

print("\n--- EXERCISE 6: Bias-Variance Tradeoff ---")
X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor

print("Depth | Train MSE | Test MSE | Interpretation")
print("-" * 55)

for depth in [1, 2, 3, 5, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))

    if depth == 1:
        interp = "Underfitting (high bias)"
    elif test_mse > train_mse * 2:
        interp = "Overfitting (high variance)"
    else:
        interp = "Good balance"

    depth_str = str(depth) if depth else "None"
    print(f"{depth_str:4s} | {train_mse:9.2f} | {test_mse:8.2f} | {interp}")

print("\n--- EXERCISE 7: Medical Diagnosis ---")
np.random.seed(42)
n_patients = 1000

X = np.random.randn(n_patients, 4)
disease_prob = 1 / (1 + np.exp(-(0.5*X[:, 0] + 0.3*X[:, 1] - 0.2*X[:, 2] + 0.01*X[:, 3])))
y = (np.random.rand(n_patients) < disease_prob).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Disease prevalence: {y.mean():.2%}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nFor medical diagnosis, RECALL is most important - we don't want to miss sick patients!")

print("\n--- EXERCISE 8: Precision-Recall ---")
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

print(f"Average Precision: {avg_precision:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'PR Curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- EXERCISE 9: LOOCV ---")
from sklearn.model_selection import LeaveOneOut

X_small = X[:50]
y_small = y[:50]

model = LogisticRegression(max_iter=1000)
loo = LeaveOneOut()

scores = []
for train_idx, test_idx in loo.split(X_small):
    X_train, X_test = X_small[train_idx], X_small[test_idx]
    y_train, y_test = y_small[train_idx], y_small[test_idx]
    model.fit(X_train, y_train)
    scores.append(model.predict(X_test) == y_test)

mean_accuracy = np.mean(scores)
print(f"LOOCV Accuracy: {mean_accuracy:.4f}")

# Compare with 5-fold
scores_5fold = cross_val_score(model, X_small, y_small, cv=5)
print(f"5-Fold CV Accuracy: {scores_5fold.mean():.4f}")

print("\n--- EXERCISE 10: Model Comparison ---")
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Naive Bayes': GaussianNB()
}

print("Model | Mean CV Accuracy | Std")
print("-" * 40)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name:22s} | {scores.mean():.4f}         | {scores.std():.4f}")
