# Phase 2: Core ML Concepts - Learning Guide

## Overview

Phase 2 introduces fundamental machine learning algorithms. You'll learn:
- **Regression**: Predict continuous values
- **Classification**: Predict categories
- **Clustering**: Group similar data
- **Evaluation**: Measure model performance

## Learning Path

### 1. Linear Regression
**What**: Predict continuous values using a linear relationship
**When**: House prices, temperature, stock prices
**Key Concepts**:
- Fitting a line to data
- Cost function (MSE)
- Gradient descent
- R² score

**Files**:
- `guide.md` - Detailed explanation
- `01_simple_regression.py` - Basic implementation
- `02_multiple_regression.py` - Multiple features
- `03_gradient_descent.py` - Algorithm explanation
- `exercises.py` - Practice problems

### 2. Logistic Regression
**What**: Predict binary outcomes (yes/no, 0/1)
**When**: Email spam, disease diagnosis, customer churn
**Key Concepts**:
- Sigmoid function
- Probability interpretation
- Decision boundary
- Log loss

**Files**:
- `guide.md` - Detailed explanation
- `01_binary_classification.py` - Basic implementation
- `02_multiclass_classification.py` - Multiple classes
- `03_probability_calibration.py` - Confidence scores
- `exercises.py` - Practice problems

### 3. Decision Trees
**What**: Tree-based model for classification and regression
**When**: Non-linear relationships, interpretability needed
**Key Concepts**:
- Information gain
- Entropy and Gini
- Tree splitting
- Overfitting prevention

**Files**:
- `guide.md` - Detailed explanation
- `01_classification_trees.py` - Classification
- `02_regression_trees.py` - Regression
- `03_tree_visualization.py` - Visualize trees
- `exercises.py` - Practice problems

### 4. Clustering
**What**: Group similar data points without labels
**When**: Customer segmentation, image compression
**Key Concepts**:
- K-Means algorithm
- Distance metrics
- Elbow method
- Silhouette score

**Files**:
- `guide.md` - Detailed explanation
- `01_kmeans_basics.py` - K-Means algorithm
- `02_clustering_evaluation.py` - Evaluate clusters
- `03_hierarchical_clustering.py` - Alternative method
- `exercises.py` - Practice problems

### 5. Model Evaluation
**What**: Measure how well models perform
**When**: Every ML project
**Key Concepts**:
- Train/test split
- Cross-validation
- Confusion matrix
- ROC-AUC, precision, recall

**Files**:
- `guide.md` - Detailed explanation
- `01_evaluation_metrics.py` - Classification metrics
- `02_regression_metrics.py` - Regression metrics
- `03_cross_validation.py` - Validation techniques
- `exercises.py` - Practice problems

## Project: Prediction Model

Build a complete prediction model:
1. Load and explore data
2. Preprocess and engineer features
3. Train multiple models
4. Evaluate and compare
5. Make predictions

**Dataset**: Iris classification or Housing regression

## Prerequisites

Before starting Phase 2, ensure you:
- ✓ Completed Phase 1 (NumPy, Pandas, Matplotlib, Statistics)
- ✓ Understand data exploration and cleaning
- ✓ Can create visualizations
- ✓ Know basic statistics

## Learning Strategy

### Week 1: Linear Regression
- Day 1-2: Understand concepts
- Day 3-4: Implement from scratch
- Day 5: Use scikit-learn
- Day 6-7: Practice with data

### Week 2: Logistic Regression
- Day 1-2: Understand concepts
- Day 3-4: Implement from scratch
- Day 5: Use scikit-learn
- Day 6-7: Practice with data

### Week 3: Decision Trees & Clustering
- Day 1-3: Decision Trees
- Day 4-5: Clustering
- Day 6-7: Practice

### Week 4: Evaluation & Project
- Day 1-3: Model evaluation
- Day 4-7: Complete project

## Key Algorithms

### Linear Regression
```
y = mx + b
Minimize: MSE = (1/n) * Σ(y_pred - y_actual)²
```

### Logistic Regression
```
P(y=1) = 1 / (1 + e^(-z))
Minimize: Log Loss = -Σ(y*log(p) + (1-y)*log(1-p))
```

### Decision Trees
```
Information Gain = Entropy(parent) - Σ(Entropy(child))
Split on feature that maximizes information gain
```

### K-Means
```
1. Initialize k centroids randomly
2. Assign points to nearest centroid
3. Update centroids
4. Repeat until convergence
```

## Common Mistakes to Avoid

1. **Not scaling features**: Different scales affect algorithms
2. **Data leakage**: Using test data in training
3. **Overfitting**: Model memorizes training data
4. **Ignoring class imbalance**: Affects classification metrics
5. **Wrong metric**: Using accuracy for imbalanced data

## Tools You'll Use

- **scikit-learn**: ML algorithms
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Seaborn**: Statistical plots

## Expected Outcomes

After Phase 2, you'll:
- Understand how ML algorithms work
- Implement algorithms from scratch
- Use scikit-learn effectively
- Evaluate model performance
- Build complete ML pipelines

## Next Steps

After Phase 2:
- Phase 3: Advanced Techniques (Ensemble, SVM, Neural Networks)
- Phase 4: Real-world Projects
- Kaggle competitions
- Your own projects

---

**Estimated Time**: 4 weeks (10-15 hours per week)

**Difficulty**: Medium - Requires understanding of Phase 1 concepts
