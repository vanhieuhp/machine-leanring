# Ensemble Methods Guide

## What are Ensemble Methods?

Ensemble methods combine multiple models to make better predictions. The idea: "Many weak learners make a strong learner."

## Why Ensemble Methods Work

- **Diversity**: Different models make different errors
- **Averaging**: Combining reduces variance
- **Boosting**: Sequential learning from mistakes
- **Stacking**: Meta-learner combines predictions

## Main Ensemble Techniques

### 1. Bagging (Bootstrap Aggregating)
- Train multiple models on random subsets
- Average predictions
- Reduces variance
- Example: Random Forest

### 2. Boosting
- Train models sequentially
- Each model learns from previous mistakes
- Reduces bias and variance
- Examples: AdaBoost, Gradient Boosting, XGBoost

### 3. Stacking
- Train multiple base models
- Train meta-model on base predictions
- Combines strengths of different models

### 4. Voting
- Train multiple models
- Majority vote (classification) or average (regression)
- Simple but effective

## Random Forest

**What**: Ensemble of decision trees

**How**:
1. Create multiple random subsets of data
2. Train decision tree on each subset
3. Average predictions

**Advantages**:
- Reduces overfitting
- Handles non-linear relationships
- Feature importance
- Parallel training

**Disadvantages**:
- Less interpretable than single tree
- Slower prediction
- Memory intensive

## Gradient Boosting

**What**: Sequential ensemble of weak learners

**How**:
1. Train first model
2. Calculate residuals
3. Train next model on residuals
4. Repeat
5. Sum all predictions

**Advantages**:
- Often best performance
- Handles complex patterns
- Feature importance

**Disadvantages**:
- Prone to overfitting
- Requires careful tuning
- Slower training

## XGBoost

**What**: Optimized gradient boosting

**Improvements**:
- Regularization (L1, L2)
- Parallel processing
- Handling missing values
- Early stopping

**When to use**:
- Kaggle competitions
- Production systems
- Large datasets

## Study Files

1. `01_random_forest.py` - Random Forest implementation
2. `02_gradient_boosting.py` - Gradient Boosting
3. `03_xgboost_advanced.py` - XGBoost advanced
4. `04_stacking.py` - Stacking ensemble
5. `exercises.py` - Practice problems
