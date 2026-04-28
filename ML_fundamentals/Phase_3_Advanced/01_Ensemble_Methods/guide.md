# 🌲 Ensemble Methods — Deep Dive Guide

## 📋 Table of Contents

1. [Overview & Intuition](#overview--intuition)
2. [Learning Roadmap](#learning-roadmap)
3. [Bagging & Random Forest](#1-bagging--random-forest)
4. [Boosting (AdaBoost, Gradient Boosting, XGBoost)](#2-boosting)
5. [Stacking & Blending](#3-stacking--blending)
6. [Voting Classifiers](#4-voting-classifiers)
7. [Hyperparameter Tuning for Ensembles](#5-hyperparameter-tuning-for-ensembles)
8. [Key Takeaways](#key-takeaways)
9. [Study Files](#study-files)

---

## Overview & Intuition

**Core Idea**: "The wisdom of the crowd" — combining predictions from multiple models often produces better results than any single model.

### Why Do Ensembles Work?

Imagine you ask 100 people to guess the number of jelly beans in a jar. Most individuals will be wrong, but the **average of all guesses** is usually very close to the true value. Ensemble methods apply this same principle to machine learning:

| Single Model Problem | Ensemble Solution |
|---|---|
| High variance (overfitting) | **Bagging** — average many models to reduce variance |
| High bias (underfitting) | **Boosting** — sequentially correct errors to reduce bias |
| Model limitations | **Stacking** — combine diverse model strengths |

### The Bias-Variance Trade-off in Ensembles

```
Total Error = Bias² + Variance + Irreducible Noise

Bagging  → primarily reduces Variance
Boosting → primarily reduces Bias (and some Variance)
Stacking → leverages diverse models to reduce both
```

---

## Learning Roadmap

Follow this step-by-step path:

### Week 1: Foundations (Days 1-4)
| Day | Topic | Study File | Time |
|-----|-------|-----------|------|
| 1 | Bagging & Random Forest concepts | `01_random_forest.py` | 2-3h |
| 2 | Random Forest hands-on | `01_random_forest.py` | 2-3h |
| 3 | AdaBoost & Gradient Boosting theory | `02_gradient_boosting.py` | 2-3h |
| 4 | Gradient Boosting hands-on | `02_gradient_boosting.py` | 2-3h |

### Week 2: Advanced (Days 5-7)
| Day | Topic | Study File | Time |
|-----|-------|-----------|------|
| 5 | XGBoost & LightGBM | `03_xgboost_advanced.py` | 2-3h |
| 6 | Stacking & Voting | `04_stacking.py` | 2-3h |
| 7 | Exercises & mini-project | `exercises.py` | 3-4h |

---

## 1. Bagging & Random Forest

### 1.1 Bootstrap Aggregating (Bagging)

**Algorithm**:
1. Create `n` **bootstrap samples** (random sampling with replacement) from the training data
2. Train a separate model on each bootstrap sample
3. Combine predictions:
   - **Classification**: majority vote
   - **Regression**: average

```
Original Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Bootstrap Sample 1: [2, 5, 5, 3, 8, 1, 10, 7, 3, 9]  → Model 1
Bootstrap Sample 2: [4, 1, 8, 8, 2, 6, 3, 10, 5, 7]  → Model 2
Bootstrap Sample 3: [7, 3, 1, 6, 9, 2, 4, 4, 8, 5]  → Model 3

Final prediction = aggregate(Model1, Model2, Model3)
```

### 1.2 Random Forest

Random Forest = Bagging + **Random Feature Selection**

**Key difference from plain Bagging**: At each split, only a **random subset of features** is considered. This adds more diversity among trees.

```
Standard Decision Tree:  considers ALL features at each split
Random Forest Tree:      considers √p features (classification) or p/3 features (regression)
    where p = total number of features
```

**Important Hyperparameters**:

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|--------------|--------|
| `n_estimators` | Number of trees | 100 - 1000 | More trees → better but slower |
| `max_depth` | Maximum tree depth | 5 - 30, or None | Deeper → more complex |
| `min_samples_split` | Min samples to split | 2 - 20 | Higher → less overfitting |
| `max_features` | Features per split | 'sqrt', 'log2', float | Lower → more diversity |
| `min_samples_leaf` | Min samples in leaf | 1 - 10 | Higher → simpler trees |

### 1.3 Out-of-Bag (OOB) Evaluation

Each bootstrap sample uses about **63.2%** of the data. The remaining **36.8%** can be used as validation — this is the **OOB score**.

```
Bootstrap Sample 1 uses: [2,5,5,3,8,1,10,7,3,9]
OOB for Sample 1:        [4, 6]  ← these weren't used for training!
```

This gives you a free validation estimate without needing a separate validation set!

### 1.4 Feature Importance

Random Forests provide feature importance scores:
- **Mean Decrease Impurity (MDI)**: How much each feature reduces impurity across all trees
- **Permutation Importance**: How much accuracy drops when a feature is randomly shuffled

> ⚠️ **Gotcha**: MDI-based importance is biased toward high-cardinality features. Prefer permutation importance for reliable results.

---

## 2. Boosting

### 2.1 AdaBoost (Adaptive Boosting)

**Core Idea**: Train weak learners sequentially. Give **more weight** to misclassified samples.

**Algorithm (step by step)**:
```
1. Initialize: all samples have equal weight = 1/N

2. For each iteration t = 1, 2, ..., T:
   a. Train weak learner h_t on weighted data
   b. Calculate weighted error: ε_t = Σ(w_i × I(h_t(x_i) ≠ y_i))
   c. Calculate learner weight: α_t = 0.5 × ln((1 - ε_t) / ε_t)
   d. Update sample weights:
      - Misclassified: w_i × exp(α_t)    ← INCREASE weight
      - Correctly classified: w_i × exp(-α_t)  ← DECREASE weight
   e. Normalize weights

3. Final prediction: H(x) = sign(Σ α_t × h_t(x))
```

**Visual intuition**:
```
Round 1: ○ ○ ○ ○ ● ● ●   (all equal weight, ● = misclassified)
Round 2: ○ ○ ○ ○ ◉ ◉ ◉   (misclassified get BIGGER weight)
Round 3: ◎ ◎ ○ ○ ● ◉ ◉   (now some previously correct get bigger)
...
Each round focuses on what the previous rounds got WRONG
```

### 2.2 Gradient Boosting

**Core Idea**: Instead of reweighting samples, fit each new model to the **residual errors** (gradients) of the previous ensemble.

**Algorithm**:
```
1. Initialize: F_0(x) = mean(y)  (start with a constant prediction)

2. For each iteration m = 1, 2, ..., M:
   a. Compute residuals: r_i = y_i - F_{m-1}(x_i)
   b. Fit a new tree h_m to residuals r_i
   c. Update model: F_m(x) = F_{m-1}(x) + η × h_m(x)
      where η = learning rate (shrinkage)

3. Final prediction: F_M(x)
```

**Numerical Example**:
```
True values:      [10.0, 15.0, 20.0]
Step 0 (mean):    [15.0, 15.0, 15.0]    prediction = mean
Residuals:        [-5.0,  0.0,  5.0]    errors to fix

Step 1 tree predicts residuals: [-4.5, 0.2, 4.8]
Updated pred (lr=0.1): [15.0 + 0.1×(-4.5), 15.0 + 0.1×0.2, 15.0 + 0.1×4.8]
                      = [14.55, 15.02, 15.48]

New residuals:    [-4.55, -0.02, 4.52]   ← smaller errors!

Step 2 tree predicts these new residuals... and so on
```

**Key Hyperparameters**:

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|--------------|--------|
| `n_estimators` | Number of boosting rounds | 100 - 5000 | More → better but risk overfitting |
| `learning_rate` | Step size (η) | 0.001 - 0.3 | Lower → needs more estimators |
| `max_depth` | Tree depth | 3 - 8 | Deeper → more complex interactions |
| `subsample` | Fraction of data per round | 0.5 - 1.0 | Lower → more regularization |
| `min_samples_leaf` | Min samples in leaf | 1 - 50 | Higher → more regularization |

> 💡 **Rule of Thumb**: Use a low learning rate (0.01-0.1) with many estimators and early stopping.

### 2.3 XGBoost (eXtreme Gradient Boosting)

**Why XGBoost over plain Gradient Boosting?**

| Feature | Gradient Boosting | XGBoost |
|---------|------------------|---------|
| Regularization | None | L1 + L2 regularization |
| Missing values | Must impute | Handles natively |
| Parallelism | Sequential | Parallel tree building |
| Speed | Slower | Column block + cache-aware |
| Early stopping | Manual | Built-in |
| Cross-validation | External | Built-in `cv()` method |

**XGBoost-specific Hyperparameters**:

| Parameter | Description | Typical Range |
|-----------|-------------|--------------|
| `reg_alpha` (α) | L1 regularization | 0 - 10 |
| `reg_lambda` (λ) | L2 regularization | 0 - 10 |
| `gamma` (γ) | Min loss reduction for split | 0 - 5 |
| `colsample_bytree` | Feature fraction per tree | 0.3 - 1.0 |
| `scale_pos_weight` | Balance for imbalanced classes | sum(neg)/sum(pos) |

### 2.4 LightGBM

LightGBM is another optimized gradient boosting framework, even faster than XGBoost on large data:

**Key Innovations**:
- **Leaf-wise growth** (vs. level-wise in XGBoost) — finds the leaf with max delta loss
- **Gradient-based One-Side Sampling (GOSS)** — keeps large-gradient instances
- **Exclusive Feature Bundling (EFB)** — bundles mutually exclusive features

```
XGBoost (level-wise):          LightGBM (leaf-wise):
       [root]                         [root]
      /      \                       /      \
   [L1]      [L2]                [L1]      [L2]
   /  \      /  \                          /  \
 [L3][L4] [L5][L6]                      [L3]  [L4]
                                              /  \
                                           [L5]  [L6]
```

---

## 3. Stacking & Blending

### 3.1 Stacking (Stacked Generalization)

**Core Idea**: Train a "meta-learner" on the predictions of multiple base models.

**Algorithm**:
```
Step 1: Split data into K folds

Step 2: For each base model (e.g., RF, SVM, KNN):
   - Train on K-1 folds, predict on the held-out fold
   - Do this for all K folds (like cross-validation)
   - This gives out-of-fold predictions for ALL training data

Step 3: Create meta-features = [RF_pred, SVM_pred, KNN_pred]

Step 4: Train meta-learner (e.g., Logistic Regression) on meta-features

Step 5: For test data:
   - Get predictions from ALL base models (trained on full training data)
   - Feed these predictions to the meta-learner
```

**Visual**:
```
Training Data
    │
    ├──► Random Forest    ──► RF predictions   ─┐
    ├──► SVM              ──► SVM predictions  ─┤──► Meta-Learner ──► Final Prediction
    ├──► KNN              ──► KNN predictions  ─┤
    └──► Gradient Boost   ──► GB predictions   ─┘
```

### 3.2 Blending

**Simpler version of stacking**: Instead of cross-validation, use a single holdout set.

```
Training Data (70%) ──► Train base models
Holdout Set   (30%) ──► Get base model predictions ──► Train meta-learner
```

**Stacking vs. Blending**:
| Aspect | Stacking | Blending |
|--------|----------|----------|
| Data usage | Efficient (K-fold) | Wastes holdout data |
| Overfitting risk | Lower | Higher |
| Complexity | More complex | Simpler |
| Speed | Slower | Faster |

---

## 4. Voting Classifiers

### 4.1 Hard Voting
Each model votes for a class. **Majority wins**.

```
Model 1: Class A
Model 2: Class B
Model 3: Class A
Model 4: Class A

Hard Vote → Class A (3 out of 4)
```

### 4.2 Soft Voting
Each model outputs **probabilities**. Average the probabilities, pick the highest.

```
Model 1: [A: 0.7, B: 0.3]
Model 2: [A: 0.4, B: 0.6]
Model 3: [A: 0.8, B: 0.2]

Average:  [A: 0.633, B: 0.367]

Soft Vote → Class A (0.633 > 0.367)
```

> 💡 **Tip**: Soft voting usually outperforms hard voting because it accounts for prediction confidence.

---

## 5. Hyperparameter Tuning for Ensembles

### Recommended Tuning Strategy

```
Step 1: Fix learning_rate = 0.1, find good n_estimators with early stopping
Step 2: Tune tree parameters (max_depth, min_samples_leaf)
Step 3: Tune regularization (subsample, colsample_bytree, reg_alpha, reg_lambda)
Step 4: Lower learning_rate (0.01-0.05), increase n_estimators proportionally
```

### Grid Search vs. Random Search vs. Bayesian

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| Grid Search | Exhaustive, deterministic | Exponential time | Few hyperparameters |
| Random Search | Faster, good coverage | Not guaranteed optimal | Moderate hyperparameters |
| Bayesian (Optuna) | Smart, efficient | Complex setup | Many hyperparameters |

---

## Key Takeaways

1. **Start with Random Forest** — it's hard to beat and easy to use
2. **Move to XGBoost/LightGBM** for maximum performance
3. **Use early stopping** to prevent overfitting in boosting
4. **Feature importance** is a valuable byproduct of tree ensembles
5. **Stacking** can squeeze out extra performance but adds complexity
6. **Always cross-validate** — don't trust single train/test split

### Decision Flowchart

```
Need a quick, strong baseline?
  └── YES → Random Forest

Need maximum performance?
  └── YES → XGBoost or LightGBM with tuning

Have very different model types?
  └── YES → Stacking or Voting

Dataset is small?
  └── YES → AdaBoost with shallow trees
```

---

## Study Files

| # | File | Description | Difficulty |
|---|------|-------------|------------|
| 1 | `01_random_forest.py` | Random Forest: from scratch concepts + sklearn | ⭐⭐ |
| 2 | `02_gradient_boosting.py` | AdaBoost, Gradient Boosting, comparison | ⭐⭐⭐ |
| 3 | `03_xgboost_advanced.py` | XGBoost, LightGBM, tuning strategies | ⭐⭐⭐ |
| 4 | `04_stacking.py` | Stacking, Blending, Voting ensembles | ⭐⭐⭐ |
| 5 | `exercises.py` | 5 hands-on exercises with solutions | ⭐⭐⭐⭐ |
