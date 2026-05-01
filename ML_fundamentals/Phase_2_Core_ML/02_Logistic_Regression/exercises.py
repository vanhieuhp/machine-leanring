"""
Logistic Regression - Exercises
================================

Practice problems for Logistic Regression.
Solutions are provided at the bottom.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, roc_curve, auc,
                           classification_report)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# ============================================================================
# EXERCISE 1: Binary Classification Basics
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Binary Classification Basics")
print("=" * 70)

# 1.1 Create sample data: study hours vs pass/fail
# Pass (1) if study hours >= 5, Fail (0) otherwise

study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
pass_fail = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

print(f"Study hours: {study_hours}")
print(f"Pass/Fail: {pass_fail}")

# 1.2 Fit logistic regression
# TODO: Fit model

# 1.3 Predict for 4.5 and 5.5 hours
# TODO: Predict

# 1.4 Get probability predictions
# TODO: Get probabilities

# 1.5 Calculate accuracy
# TODO: Calculate accuracy

# ============================================================================
# EXERCISE 2: Confusion Matrix
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Confusion Matrix")
print("=" * 70)

# 2.1 Create imbalanced dataset
# 90% class 0, 10% class 1
X_imbalance, y_imbalance = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    weights=[0.9, 0.1], random_state=42
)

# 2.2 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imbalance, y_imbalance, test_size=0.2, random_state=42
)

# 2.3 Train logistic regression
# TODO: Train

# 2.4 Make predictions
# TODO: Predict

# 2.5 Calculate confusion matrix
# TODO: Calculate

# 2.6 Calculate: TP, TN, FP, FN
# TODO: Extract values

# 2.7 Calculate: Accuracy, Precision, Recall, F1
# TODO: Calculate each metric

# ============================================================================
# EXERCISE 3: Multi-class Classification
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Multi-class Classification")
print("=" * 70)

# 3.1 Create 3-class dataset
# Features: exam1_score, exam2_score
# Classes: Low, Medium, High pass probability

np.random.seed(42)
n = 150

# Low: both exams < 50
X_low = np.random.uniform(30, 50, (50, 2))
y_low = np.zeros(50)

# Medium: one exam 50-70
X_med = np.random.uniform(40, 70, (50, 2))
y_med = np.ones(50)

# High: both exams > 70
X_high = np.random.uniform(60, 100, (50, 2))
y_high = np.ones(50) * 2

X_multi = np.vstack([X_low, X_med, X_high])
y_multi = np.hstack([y_low, y_med, y_high])

# Shuffle
idx = np.random.permutation(len(y_multi))
X_multi = X_multi[idx]
y_multi = y_multi[idx]

# 3.2 Split data
# TODO: Split

# 3.3 Train logistic regression with multi_class='multinomial'
# TODO: Train

# 3.4 Predict for new student scores: (60, 65)
# TODO: Predict

# 3.5 Calculate accuracy
# TODO: Calculate accuracy

# ============================================================================
# EXERCISE 4: ROC Curve and AUC
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: ROC Curve and AUC")
print("=" * 70)

# 4.1 Create dataset
X_roc, y_roc = make_classification(n_samples=200, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_roc, y_roc, test_size=0.2, random_state=42)

# 4.2 Train model
# TODO: Train

# 4.3 Get probability predictions
# TODO: Get probabilities

# 4.4 Calculate ROC curve
# TODO: Calculate

# 4.5 Calculate AUC score
# TODO: Calculate

# 4.6 Plot ROC curve
# TODO: Plot

# ============================================================================
# EXERCISE 5: Threshold Tuning
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Threshold Tuning")
print("=" * 70)

# 5.1 Using same data from Exercise 4
# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# 5.2 Try different thresholds: 0.3, 0.5, 0.7
thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    # TODO: Predict with threshold

    # TODO: Calculate metrics

    print(f"\nThreshold: {threshold}")
    # TODO: Print accuracy, precision, recall

# 5.3 Which threshold is best for:
# - High precision? (minimize FP)
# - High recall? (minimize FN)
# - Balanced? (F1)

# ============================================================================
# EXERCISE 6: Regularization
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Regularization")
print("=" * 70)

from sklearn.linear_model import LogisticRegression

# 6.1 Create data with many features (some noise)
X_reg, y_reg = make_classification(n_samples=200, n_features=20,
                                   n_informative=5, n_redundant=15,
                                   random_state=42)

# 6.2 Train with different C values (regularization strength)
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

for C in C_values:
    # TODO: Train with different C

    # TODO: Calculate accuracy

    print(f"C={C}: Accuracy={accuracy:.4f}")

# 6.3 Which C gives best result?
# Hint: Smaller C = stronger regularization

# ============================================================================
# EXERCISE 7: Real-world Problem - Email Spam Detection
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Email Spam Detection")
print("=" * 70)

# Simulated email data
# Features: word_count, link_count, capital_letters_ratio
np.random.seed(42)
n_emails = 200

# Not spam
word_count_nospam = np.random.normal(150, 50, n_emails // 2)
link_count_nospam = np.random.normal(2, 1, n_emails // 2)
capital_nospam = np.random.normal(0.1, 0.05, n_emails // 2)
y_nospam = np.zeros(n_emails // 2)

# Spam
word_count_spam = np.random.normal(300, 100, n_emails // 2)
link_count_spam = np.random.normal(10, 3, n_emails // 2)
capital_spam = np.random.normal(0.3, 0.1, n_emails // 2)
y_spam = np.ones(n_emails // 2)

# Combine
emails = pd.DataFrame({
    'word_count': np.hstack([word_count_nospam, word_count_spam]),
    'link_count': np.hstack([link_count_nospam, link_count_spam]),
    'capital_ratio': np.hstack([capital_nospam, capital_spam]),
    'is_spam': np.hstack([y_nospam, y_spam])
})

# Shuffle
emails = emails.sample(frac=1).reset_index(drop=True)

print("Email Dataset:")
print(emails.head())

# 7.1 Split features and target
# TODO: X, y

# 7.2 Split train/test
# TODO: Split

# 7.3 Train logistic regression
# TODO: Train

# 7.4 Evaluate
# TODO: Predict and evaluate

# 7.5 Predict for new email:
# - 250 words, 8 links, 25% capital letters
new_email = [[250, 8, 0.25]]
# TODO: Predict

# ============================================================================
# EXERCISE 8: Sigmoid Function Implementation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Sigmoid Function Implementation")
print("=" * 70)

# 8.1 Implement sigmoid function
def sigmoid(z):
    """
    Implement sigmoid function.

    Args:
        z: Input value(s)

    Returns:
        Sigmoid of z (between 0 and 1)
    """
    # TODO: Implement
    pass

# Test
test_values = [-10, -5, 0, 5, 10]
print("Sigmoid function test:")
for z in test_values:
    # TODO: Calculate and print

# 8.2 Implement logistic regression prediction
def predict_proba(X, weights, bias):
    """
    Predict probability using logistic regression.

    Args:
        X: Features (n_samples, n_features)
        weights: Model weights
        bias: Model bias

    Returns:
        Probability of class 1
    """
    # TODO: Implement
    # Hint: z = X @ weights + bias, then apply sigmoid
    pass

# Test with simple data
X_test = np.array([[1, 2], [3, 4], [5, 6]])
weights_test = np.array([0.5, -0.3])
bias_test = 0.1

# TODO: Predict

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("\n--- EXERCISE 1: Binary Classification Basics ---")
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
pass_fail = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(study_hours.reshape(-1, 1), pass_fail)

print(f"Model coefficients: {model.coef_[0]:.4f}, intercept: {model.intercept_[0]:.4f}")
print(f"Predict for 4.5 hours: {model.predict([[4.5]])[0]}")
print(f"Predict for 5.5 hours: {model.predict([[5.5]])[0]}")
print(f"Probabilities: {model.predict_proba([[4.5], [5.5]])}")
print(f"Accuracy: {accuracy_score(pass_fail, model.predict(study_hours.reshape(-1, 1))):.4f}")

print("\n--- EXERCISE 2: Confusion Matrix ---")
X_imbalance, y_imbalance = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                               weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_imbalance, y_imbalance, test_size=0.2, random_state=42)

model = LogisticRegression()
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

print("\n--- EXERCISE 3: Multi-class Classification ---")
np.random.seed(42)
n = 150
X_low = np.random.uniform(30, 50, (50, 2))
X_med = np.random.uniform(40, 70, (50, 2))
X_high = np.random.uniform(60, 100, (50, 2))
y_low, y_med, y_high = np.zeros(50), np.ones(50), np.ones(50) * 2

X_multi = np.vstack([X_low, X_med, X_high])
y_multi = np.hstack([y_low, y_med, y_high])
idx = np.random.permutation(len(y_multi))
X_multi, y_multi = X_multi[idx], y_multi[idx]

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)

print(f"Prediction for (60, 65): {model.predict([[60, 65]])[0]}")
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

print("\n--- EXERCISE 4: ROC Curve and AUC ---")
X_roc, y_roc = make_classification(n_samples=200, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_roc, y_roc, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC: {roc_auc:.4f}")

print("\n--- EXERCISE 5: Threshold Tuning ---")
y_proba = model.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.5, 0.7]:
    y_pred = (y_proba >= threshold).astype(int)
    print(f"Threshold: {threshold}")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.4f}")

print("\n--- EXERCISE 6: Regularization ---")
X_reg, y_reg = make_classification(n_samples=200, n_features=20,
                                   n_informative=5, n_redundant=15,
                                   random_state=42)

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_reg, y_reg)
    acc = accuracy_score(y_reg, model.predict(X_reg))
    print(f"C={C}: Accuracy={acc:.4f}")

print("\n--- EXERCISE 7: Email Spam Detection ---")
np.random.seed(42)
n_emails = 200
word_count_nospam = np.random.normal(150, 50, n_emails // 2)
link_count_nospam = np.random.normal(2, 1, n_emails // 2)
capital_nospam = np.random.normal(0.1, 0.05, n_emails // 2)
y_nospam = np.zeros(n_emails // 2)

word_count_spam = np.random.normal(300, 100, n_emails // 2)
link_count_spam = np.random.normal(10, 3, n_emails // 2)
capital_spam = np.random.normal(0.3, 0.1, n_emails // 2)
y_spam = np.ones(n_emails // 2)

emails = pd.DataFrame({
    'word_count': np.hstack([word_count_nospam, word_count_spam]),
    'link_count': np.hstack([link_count_nospam, link_count_spam]),
    'capital_ratio': np.hstack([capital_nospam, capital_spam]),
    'is_spam': np.hstack([y_nospam, y_spam])
})
emails = emails.sample(frac=1).reset_index(drop=True)

X = emails[['word_count', 'link_count', 'capital_ratio']]
y = emails['is_spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(f"Prediction for new email: {model.predict([[250, 8, 0.25]])[0]}")
print(f"Probability: {model.predict_proba([[250, 8, 0.25]])}")

print("\n--- EXERCISE 8: Sigmoid Function ---")
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, weights, bias):
    z = X @ weights + bias
    return sigmoid(z)

X_test = np.array([[1, 2], [3, 4], [5, 6]])
weights_test = np.array([0.5, -0.3])
bias_test = 0.1

probas = predict_proba(X_test, weights_test, bias_test)
print(f"Test predictions: {probas}")

print("\nSigmoid tests:")
for z in [-10, -5, 0, 5, 10]:
    print(f"sigmoid({z}) = {sigmoid(z):.4f}")
