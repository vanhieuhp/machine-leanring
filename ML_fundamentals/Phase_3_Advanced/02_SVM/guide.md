# Support Vector Machines Guide

## What is SVM?

Support Vector Machines find the optimal hyperplane that maximizes the margin between classes.

## Key Concepts

### 1. Hyperplane
- Decision boundary separating classes
- In 2D: line, in 3D: plane, in nD: hyperplane

### 2. Margin
- Distance from hyperplane to nearest points
- Larger margin = better generalization

### 3. Support Vectors
- Data points closest to hyperplane
- Define the margin
- Only these matter for prediction

### 4. Kernel Trick
- Transform data to higher dimension
- Make non-linear problems linear
- Common kernels: linear, RBF, polynomial

## Kernels

**Linear**: For linearly separable data
```
K(x, y) = x · y
```

**RBF (Radial Basis Function)**: For non-linear data
```
K(x, y) = exp(-γ||x - y||²)
```

**Polynomial**: For polynomial relationships
```
K(x, y) = (x · y + c)^d
```

## Advantages

- Works well in high dimensions
- Memory efficient (only support vectors)
- Effective with non-linear kernels
- Good for binary classification

## Disadvantages

- Slow for large datasets
- Requires feature scaling
- Hard to interpret
- Hyperparameter tuning needed

## When to Use

- Binary classification
- Small to medium datasets
- High-dimensional data
- When interpretability not critical

## Study Files

1. `01_svm_basics.py` - SVM fundamentals
2. `02_kernels.py` - Different kernels
3. `03_multiclass_svm.py` - Multi-class SVM
4. `04_svm_regression.py` - SVM for regression
5. `exercises.py` - Practice problems
