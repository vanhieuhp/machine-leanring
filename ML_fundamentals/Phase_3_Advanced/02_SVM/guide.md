# 🎯 Support Vector Machines — Deep Dive Guide

## 📋 Table of Contents

1. [Overview & Intuition](#overview--intuition)
2. [Learning Roadmap](#learning-roadmap)
3. [Linear SVM](#1-linear-svm)
4. [Kernel Trick](#2-the-kernel-trick)
5. [SVM for Multi-class](#3-svm-for-multi-class-classification)
6. [SVM for Regression (SVR)](#4-svm-for-regression-svr)
7. [Hyperparameter Tuning](#5-hyperparameter-tuning)
8. [Key Takeaways](#key-takeaways)

---

## Overview & Intuition

**Core Idea**: Find the **optimal hyperplane** that separates classes with the **maximum margin**.

### Visual Intuition (2D)

```
  Class B (+)         ─── Hyperplane ───         Class A (○)
                     ╱                  ╲               
    +   +          ╱   MARGIN (maximize) ╲         ○   ○
      +    +     ╱←─── support vector ───→╲    ○  ○
    +   +      ╱                            ╲    ○   ○
      +      ╱   ●                      ●    ╲  ○
    +      ╱    (sv)                    (sv)   ╲   ○
         ╱                                      ╲
```

### Why Maximum Margin?

| Small Margin | Large Margin |
|---|---|
| Sensitive to noise | Robust to noise |
| High variance | Low variance |
| Overfits | Generalizes better |

### Key Terms

| Term | Definition |
|------|------------|
| **Hyperplane** | Decision boundary (line in 2D, plane in 3D, hyperplane in nD) |
| **Margin** | Distance between hyperplane and nearest data points |
| **Support Vectors** | Data points closest to the hyperplane (they "support" it) |
| **Kernel** | Function that maps data to higher dimensions |
| **C** | Regularization parameter (trade-off: margin width vs. misclassification) |
| **γ (gamma)** | Kernel coefficient for RBF kernel (reach of influence) |

---

## Learning Roadmap

| Day | Topic | Study File | Time |
|-----|-------|-----------|------|
| 1 | Linear SVM, margins, C parameter | `01_svm_basics.py` | 2-3h |
| 2 | Kernel trick (RBF, polynomial) | `02_kernels.py` | 2-3h |
| 3 | Multi-class SVM, One-vs-One, One-vs-Rest | `03_multiclass_svm.py` | 2h |
| 4 | SVR (regression) + tuning | `04_svm_regression.py` | 2-3h |
| 5 | Exercises & practice | `exercises.py` | 3h |

---

## 1. Linear SVM

### 1.1 The Optimization Problem

SVM solves this optimization:

```
Minimize:    ½||w||² + C × Σ ξ_i

Subject to:  y_i(w·x_i + b) ≥ 1 - ξ_i
             ξ_i ≥ 0
```

Where:
- `w` = weight vector (defines hyperplane direction)
- `b` = bias (shifts hyperplane)
- `ξ_i` = slack variables (allow misclassification)
- `C` = regularization (penalty for misclassification)

### 1.2 Hard Margin vs Soft Margin

```
Hard Margin (C = ∞):              Soft Margin (C = finite):
  No misclassification allowed      Some misclassification allowed
  Only works if linearly separable  Works with noisy/overlapping data
  Will overfit to outliers          More robust, generalizes better

  + + +  |  ○ ○ ○               + + +  | ○ ○ ○
  + + +  |  ○ ○ ○               + + ○  | ○ ○ ○  ← allowed!
  + + +  |  ○ ○ ○               + + +  | + ○ ○  ← allowed!
```

### 1.3 The C Parameter

```
C = 0.001 (very small)     C = 1.0 (moderate)      C = 1000 (very large)
────────────────────────    ────────────────────    ────────────────────
Wide margin                 Balanced margin          Narrow margin
Many misclassifications     Few misclassifications   Almost no misclass.
Underfitting risk           Good generalization      Overfitting risk
High bias, low variance     Balanced                 Low bias, high var.
```

> 💡 **Rule of Thumb**: Start with C=1.0, then tune with cross-validation.

---

## 2. The Kernel Trick

### Why Kernels?

When data isn't linearly separable in the original space, we can project it to a **higher-dimensional space** where it becomes separable.

```
Original 2D Space:               After RBF Kernel (higher dim):
                                  
  ○ ○ + ○ ○                     ○ ○       ○ ○
  ○ + + + ○                       ○         ○
  + + + + +      ──kernel──→            +
  ○ + + + ○                       ○   + + +   ○
  ○ ○ + ○ ○                     ○ ○  + + +  ○ ○
                                  (now linearly separable!)
Not linearly separable!
```

### Common Kernels

| Kernel | Formula | When to Use |
|--------|---------|-------------|
| **Linear** | K(x,y) = x·y | Linearly separable, high-dimensional, text data |
| **RBF** | K(x,y) = exp(-γ‖x-y‖²) | Non-linear data (default, most common) |
| **Polynomial** | K(x,y) = (γ·x·y + r)^d | Polynomial relationships |
| **Sigmoid** | K(x,y) = tanh(γ·x·y + r) | Similar to neural networks |

### Gamma (γ) Parameter (for RBF)

```
γ small (0.001)        γ moderate (0.1)       γ large (10)
──────────────────    ──────────────────    ──────────────────
Each point has         Moderate reach        Each point only
wide influence                               affects neighbors
Smooth boundary        Good boundary         Very complex boundary
Underfitting           Good generalization   Overfitting
```

---

## 3. SVM for Multi-class Classification

SVM is natively a **binary** classifier. For multi-class, two strategies:

### One-vs-Rest (OvR)

```
3 classes → 3 binary classifiers:
  Classifier 1: Class A vs (B + C)
  Classifier 2: Class B vs (A + C)
  Classifier 3: Class C vs (A + B)

Prediction: class with highest confidence score
```

### One-vs-One (OvO)

```
3 classes → 3 binary classifiers:
  Classifier 1: Class A vs Class B
  Classifier 2: Class A vs Class C
  Classifier 3: Class B vs Class C

Prediction: majority vote among classifiers
n_classifiers = n_classes × (n_classes - 1) / 2
```

| Aspect | One-vs-Rest | One-vs-One |
|--------|-------------|------------|
| # classifiers | n_classes | n×(n-1)/2 |
| Training data each | All | Subset (2 classes) |
| sklearn SVC default | ❌ | ✅ |
| sklearn LinearSVC | ✅ | ❌ |

---

## 4. SVM for Regression (SVR)

Instead of finding a margin that separates classes, SVR finds a tube (ε-tube) that contains most data points.

```
        _______________
       /  ε-tube       \
  ────/──────────────────\──── regression line
     / ● ●  ●  ●   ●  ●  \
    /   ●             ●    \
   ──────────────────────────

  Points inside ε-tube: no penalty
  Points outside: penalized proportional to distance
```

Key parameter: **ε (epsilon)** — width of the tube

---

## 5. Hyperparameter Tuning

### Tuning Strategy

```
Step 1: Choose kernel (start with RBF)
Step 2: Scale features! (CRITICAL for SVM)
Step 3: Tune C and gamma together (GridSearchCV)
Step 4: Evaluate with cross-validation
```

### Common Parameter Ranges

| Parameter | Range to Try |
|-----------|-------------|
| C | [0.001, 0.01, 0.1, 1, 10, 100, 1000] |
| gamma | ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10] |
| kernel | ['linear', 'rbf', 'poly'] |
| degree (poly) | [2, 3, 4, 5] |

> ⚠️ **CRITICAL**: Always scale features before SVM! Use `StandardScaler` or `Pipeline`.

---

## Key Takeaways

1. **Always scale features** — SVM is distance-based
2. **Start with RBF kernel** — it's the most versatile
3. **C controls regularization** — higher C = less regularization
4. **γ controls decision boundary complexity** — higher γ = more complex
5. **SVM works best on small-medium datasets** — slow on large datasets
6. **Support vectors define the model** — most training points are irrelevant
7. **Use Pipeline** to combine scaling + SVM

### When to Use SVM

```
✅ Good for:                       ❌ Avoid when:
  • Binary classification           • Dataset is very large (>100K)
  • High-dimensional data            • Need probability estimates
  • Small-medium datasets            • Need interpretability
  • Clear margin of separation       • Many noisy features
  • Text classification (linear)     • Online learning needed
```

---

## Study Files

| # | File | Description | Difficulty |
|---|------|-------------|------------|
| 1 | `01_svm_basics.py` | Linear SVM, margins, C parameter | ⭐⭐ |
| 2 | `02_kernels.py` | RBF, Poly, kernel comparison | ⭐⭐⭐ |
| 3 | `03_multiclass_svm.py` | OvO, OvR, multi-class strategies | ⭐⭐ |
| 4 | `04_svm_regression.py` | SVR, epsilon-tube, tuning | ⭐⭐⭐ |
| 5 | `exercises.py` | 5 practice problems with solutions | ⭐⭐⭐ |
