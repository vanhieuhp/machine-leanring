"""
Logistic Regression - Part 1: Binary Classification
====================================================

This module covers:
- Sigmoid function
- Binary classification
- Decision boundary
- Probability predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. SIGMOID FUNCTION
# ============================================================================

print("=" * 70)
print("1. SIGMOID FUNCTION")
print("=" * 70)

def sigmoid(z):
    """Sigmoid function: 1 / (1 + e^-z)"""
    return 1 / (1 + np.exp(-z))

# Plot sigmoid
z = np.linspace(-10, 10, 100)
y = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, linewidth=2, color='blue')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision boundary (0.5)')
plt.axvline(x=0, color='green', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("Sigmoid properties:")
print(f"  σ(-10) = {sigmoid(-10):.6f}")
print(f"  σ(0) = {sigmoid(0):.6f}")
print(f"  σ(10) = {sigmoid(10):.6f}")

# ============================================================================
# 2. BINARY CLASSIFICATION EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("2. BINARY CLASSIFICATION EXAMPLE")
print("=" * 70)

# Generate sample data
np.random.seed(42)
n_samples = 200

# Class 0
X0 = np.random.randn(n_samples // 2, 2) + np.array([0, 0])
y0 = np.zeros(n_samples // 2)

# Class 1
X1 = np.random.randn(n_samples // 2, 2) + np.array([3, 3])
y1 = np.ones(n_samples // 2)

# Combine
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

print(f"Data shape: {X.shape}")
print(f"Class 0: {np.sum(y == 0)} samples")
print(f"Class 1: {np.sum(y == 1)} samples")

# ============================================================================
# 3. TRAIN LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("3. TRAIN LOGISTIC REGRESSION")
print("=" * 70)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Coefficients
print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# ============================================================================
# 4. PREDICTIONS AND PROBABILITIES
# ============================================================================

print("\n" + "=" * 70)
print("4. PREDICTIONS AND PROBABILITIES")
print("=" * 70)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("First 10 predictions:")
for i in range(10):
    print(f"  Sample {i}: P(y=0)={y_pred_proba[i, 0]:.4f}, P(y=1)={y_pred_proba[i, 1]:.4f}, Prediction={y_pred[i]:.0f}")

# ============================================================================
# 5. CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 70)
print("5. CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# ============================================================================
# 6. CLASSIFICATION METRICS
# ============================================================================

print("\n" + "=" * 70)
print("6. CLASSIFICATION METRICS")
print("=" * 70)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# ============================================================================
# 7. ROC CURVE
# ============================================================================

print("\n" + "=" * 70)
print("7. ROC CURVE")
print("=" * 70)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

print(f"ROC AUC Score: {roc_auc:.4f}")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("8. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Data and decision boundary
ax = axes[0, 0]
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], alpha=0.6, label='Class 0', color='blue')
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], alpha=0.6, label='Class 1', color='red')

# Decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Decision Boundary')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Confusion Matrix
ax = axes[0, 1]
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
ax.set_yticklabels(['Actual 0', 'Actual 1'])
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=16)
plt.colorbar(im, ax=ax)

# Plot 3: ROC Curve
ax = axes[1, 0]
ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Probability Distribution
ax = axes[1, 1]
ax.hist(y_pred_proba[y_test == 0, 1], bins=20, alpha=0.6, label='Class 0', color='blue')
ax.hist(y_pred_proba[y_test == 1, 1], bins=20, alpha=0.6, label='Class 1', color='red')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
ax.set_xlabel('Predicted Probability of Class 1')
ax.set_ylabel('Frequency')
ax.set_title('Probability Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 9. PRACTICAL EXAMPLE: DISEASE DIAGNOSIS
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL EXAMPLE: DISEASE DIAGNOSIS")
print("=" * 70)

# Simulate medical data
np.random.seed(42)
n_patients = 300

# Healthy patients
age_healthy = np.random.normal(40, 15, n_patients // 2)
blood_pressure_healthy = np.random.normal(120, 10, n_patients // 2)
y_healthy = np.zeros(n_patients // 2)

# Sick patients
age_sick = np.random.normal(55, 15, n_patients // 2)
blood_pressure_sick = np.random.normal(150, 15, n_patients // 2)
y_sick = np.ones(n_patients // 2)

X_medical = np.vstack([
    np.column_stack([age_healthy, blood_pressure_healthy]),
    np.column_stack([age_sick, blood_pressure_sick])
])
y_medical = np.hstack([y_healthy, y_sick])

# Train model
X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_medical, y_medical, test_size=0.2, random_state=42
)

model_medical = LogisticRegression(random_state=42)
model_medical.fit(X_train_med, y_train_med)

# Evaluate
y_pred_med = model_medical.predict(X_test_med)
accuracy_med = accuracy_score(y_test_med, y_pred_med)

print(f"Medical Model Accuracy: {accuracy_med:.4f}")

# Predict for new patient
new_patient = np.array([[50, 140]])
prediction = model_medical.predict(new_patient)[0]
probability = model_medical.predict_proba(new_patient)[0, 1]

print(f"\nNew patient: Age=50, Blood Pressure=140")
print(f"Prediction: {'Sick' if prediction == 1 else 'Healthy'}")
print(f"Probability of disease: {probability:.4f}")
