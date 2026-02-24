"""
Model Evaluation - Part 1: Classification Metrics
==================================================

This module covers:
- Confusion matrix
- Accuracy, Precision, Recall, F1
- ROC-AUC
- Cross-validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                            recall_score, f1_score, roc_curve, auc, roc_auc_score)
from sklearn.datasets import make_classification

# ============================================================================
# 1. CONFUSION MATRIX EXPLAINED
# ============================================================================

print("=" * 70)
print("1. CONFUSION MATRIX EXPLAINED")
print("=" * 70)

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix:")
print(f"  True Negatives (TN): {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP): {tp}")

# ============================================================================
# 2. CLASSIFICATION METRICS
# ============================================================================

print("\n" + "=" * 70)
print("2. CLASSIFICATION METRICS")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"  Formula: (TP + TN) / Total")
print(f"  Calculation: ({tp} + {tn}) / {len(y_test)} = {accuracy:.4f}")

print(f"\nPrecision: {precision:.4f}")
print(f"  Formula: TP / (TP + FP)")
print(f"  Calculation: {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"  Meaning: Of predicted positive, how many are correct?")

print(f"\nRecall: {recall:.4f}")
print(f"  Formula: TP / (TP + FN)")
print(f"  Calculation: {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"  Meaning: Of actual positive, how many did we find?")

print(f"\nF1-Score: {f1:.4f}")
print(f"  Formula: 2 * (Precision * Recall) / (Precision + Recall)")
print(f"  Calculation: 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
print(f"  Meaning: Harmonic mean of precision and recall")

# ============================================================================
# 3. WHEN TO USE WHICH METRIC
# ============================================================================

print("\n" + "=" * 70)
print("3. WHEN TO USE WHICH METRIC")
print("=" * 70)

print("""
Accuracy:
  - Use when: Classes are balanced
  - Avoid when: Classes are imbalanced
  - Example: 95% accuracy on 95% negative class is useless

Precision:
  - Use when: False positives are costly
  - Example: Email spam detection (don't want to block real emails)
  - High precision = few false alarms

Recall:
  - Use when: False negatives are costly
  - Example: Disease diagnosis (don't want to miss sick patients)
  - High recall = catch all positive cases

F1-Score:
  - Use when: Need balance between precision and recall
  - Example: General classification tasks
  - Good for imbalanced datasets
""")

# ============================================================================
# 4. ROC-AUC CURVE
# ============================================================================

print("\n" + "=" * 70)
print("4. ROC-AUC CURVE")
print("=" * 70)

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Interpretation:")
print(f"  0.5 = Random classifier")
print(f"  1.0 = Perfect classifier")
print(f"  {roc_auc:.4f} = {'Good' if roc_auc > 0.7 else 'Fair' if roc_auc > 0.6 else 'Poor'} classifier")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("5. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Confusion Matrix
ax = axes[0, 0]
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

# Plot 2: Metrics comparison
ax = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['blue', 'green', 'orange', 'red']
ax.bar(metrics, values, color=colors, alpha=0.7)
ax.set_ylabel('Score')
ax.set_title('Classification Metrics')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

# Plot 3: ROC Curve
ax = axes[1, 0]
ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Threshold analysis
ax = axes[1, 1]
thresholds_plot = np.linspace(0, 1, 100)
precisions = []
recalls = []

for threshold in thresholds_plot:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    if len(np.unique(y_pred_threshold)) > 1:
        precisions.append(precision_score(y_test, y_pred_threshold, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_threshold, zero_division=0))
    else:
        precisions.append(0)
        recalls.append(0)

ax.plot(thresholds_plot, precisions, label='Precision', linewidth=2)
ax.plot(thresholds_plot, recalls, label='Recall', linewidth=2)
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision vs Recall vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("6. CROSS-VALIDATION")
print("=" * 70)

# K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

print(f"5-Fold Cross-Validation F1 Scores:")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}: {score:.4f}")

print(f"\nMean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 7. PRACTICAL EXAMPLE: FRAUD DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL EXAMPLE: FRAUD DETECTION")
print("=" * 70)

# Simulate fraud data (highly imbalanced)
X_fraud, y_fraud = make_classification(n_samples=10000, n_features=20, n_informative=10,
                                       n_redundant=5, weights=[0.99, 0.01], random_state=42)

X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42
)

# Train model
model_fraud = LogisticRegression(random_state=42, max_iter=1000)
model_fraud.fit(X_train_fraud, y_train_fraud)
y_pred_fraud = model_fraud.predict(X_test_fraud)

# Evaluate
accuracy_fraud = accuracy_score(y_test_fraud, y_pred_fraud)
precision_fraud = precision_score(y_test_fraud, y_pred_fraud)
recall_fraud = recall_score(y_test_fraud, y_pred_fraud)
f1_fraud = f1_score(y_test_fraud, y_pred_fraud)

print(f"Fraud Detection Model:")
print(f"  Accuracy: {accuracy_fraud:.4f}")
print(f"  Precision: {precision_fraud:.4f}")
print(f"  Recall: {recall_fraud:.4f}")
print(f"  F1-Score: {f1_fraud:.4f}")

print(f"\nInterpretation:")
print(f"  High accuracy ({accuracy_fraud:.4f}) is misleading (mostly predicting no fraud)")
print(f"  Precision ({precision_fraud:.4f}): When we predict fraud, we're correct {precision_fraud*100:.1f}% of the time")
print(f"  Recall ({recall_fraud:.4f}): We catch {recall_fraud*100:.1f}% of actual fraud cases")
print(f"  F1-Score ({f1_fraud:.4f}): Better metric for imbalanced data")
