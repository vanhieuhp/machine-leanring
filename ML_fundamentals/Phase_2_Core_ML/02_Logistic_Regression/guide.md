# Logistic Regression Guide

## What is Logistic Regression?

Logistic regression predicts binary outcomes (yes/no, 0/1, true/false). Despite its name, it's a classification algorithm, not regression.

## When to Use

- Email spam detection (spam/not spam)
- Disease diagnosis (disease/no disease)
- Customer churn (will leave/will stay)
- Loan approval (approve/deny)

## Key Concepts

### 1. Sigmoid Function
```
P(y=1) = 1 / (1 + e^(-z))
```
- Converts any value to probability (0 to 1)
- S-shaped curve
- Smooth transition at 0.5

### 2. Decision Boundary
- Default threshold: 0.5
- If P(y=1) > 0.5: predict class 1
- If P(y=1) ≤ 0.5: predict class 0

### 3. Log Loss (Binary Cross-Entropy)
```
Loss = -Σ(y*log(p) + (1-y)*log(1-p))
```
- Penalizes confident wrong predictions
- Minimized during training

### 4. Multiclass Classification
- One-vs-Rest: Train k binary classifiers
- Softmax: Generalization of sigmoid

## Evaluation Metrics

**Confusion Matrix**:
- True Positives (TP): Correctly predicted positive
- True Negatives (TN): Correctly predicted negative
- False Positives (FP): Incorrectly predicted positive
- False Negatives (FN): Incorrectly predicted negative

**Metrics**:
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## Advantages

- Probabilistic predictions
- Interpretable coefficients
- Fast to train
- Works well for linearly separable data

## Disadvantages

- Assumes linear decision boundary
- Sensitive to outliers
- Requires feature scaling
- Can underfit complex data

## Study Files

1. `01_binary_classification.py` - Binary classification
2. `02_multiclass_classification.py` - Multiple classes
3. `03_probability_calibration.py` - Confidence scores
4. `exercises.py` - Practice problems
