# Decision Trees Guide

## What is a Decision Tree?

A decision tree is a tree-like model that makes decisions by splitting data based on feature values. It's interpretable and works for both classification and regression.

## When to Use

- Customer segmentation
- Medical diagnosis
- Credit approval
- Feature importance analysis

## Key Concepts

### 1. Tree Structure
- **Root**: Starting node with all data
- **Internal nodes**: Decision points (splits)
- **Leaves**: Final predictions
- **Branches**: Outcomes of decisions

### 2. Splitting Criteria

**Information Gain** (Classification):
```
IG = Entropy(parent) - Σ(Entropy(child))
```

**Gini Impurity**:
```
Gini = 1 - Σ(p_i²)
```

**Variance Reduction** (Regression):
```
VR = Var(parent) - Σ(Var(child))
```

### 3. Stopping Criteria

- Maximum depth reached
- Minimum samples in leaf
- No improvement in split
- Pure node (all same class)

## Advantages

- Interpretable and visual
- Handles non-linear relationships
- No feature scaling needed
- Works with categorical data

## Disadvantages

- Prone to overfitting
- Unstable (small data changes = big tree changes)
- Biased toward high-cardinality features
- Can create very deep trees

## Study Files

1. `01_classification_trees.py` - Classification
2. `02_regression_trees.py` - Regression
3. `03_tree_visualization.py` - Visualize trees
4. `exercises.py` - Practice problems
