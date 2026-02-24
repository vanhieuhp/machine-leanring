# Model Evaluation Guide

## Why Evaluation Matters

Proper evaluation ensures your model:
- Generalizes to new data
- Doesn't overfit or underfit
- Meets business requirements
- Is better than baseline

## Key Concepts

### 1. Train/Test Split

**Problem**: Training on test data leads to overfitting

**Solution**: Split data:
- Training set: 70-80% (train model)
- Test set: 20-30% (evaluate model)

### 2. Cross-Validation

**k-Fold Cross-Validation**:
1. Split data into k folds
2. Train on k-1 folds, test on 1 fold
3. Repeat k times
4. Average results

**Advantages**:
- Uses all data for training and testing
- More reliable estimate
- Detects overfitting

### 3. Classification Metrics

**Confusion Matrix**:
```
                Predicted
              Positive  Negative
Actual Positive   TP       FN
       Negative   FP       TN
```

**Metrics**:
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN) - Overall correctness
- **Precision**: TP/(TP+FP) - Of predicted positive, how many correct
- **Recall**: TP/(TP+FN) - Of actual positive, how many found
- **F1-Score**: 2*(P*R)/(P+R) - Harmonic mean of precision and recall

### 4. Regression Metrics

- **MAE**: Mean Absolute Error - Average absolute difference
- **MSE**: Mean Squared Error - Average squared difference
- **RMSE**: Root Mean Squared Error - Same units as target
- **R²**: Coefficient of determination - Proportion of variance explained

### 5. ROC-AUC

- **ROC Curve**: Plot TPR vs FPR at different thresholds
- **AUC**: Area Under Curve (0 to 1)
- **Interpretation**: 0.5 = random, 1.0 = perfect

## Common Mistakes

1. **Evaluating on training data**: Always use separate test set
2. **Using wrong metric**: Accuracy bad for imbalanced data
3. **Not checking baseline**: Compare to simple model
4. **Ignoring class imbalance**: Use stratified split
5. **Overfitting to test set**: Use validation set for tuning

## Study Files

1. `01_evaluation_metrics.py` - Classification metrics
2. `02_regression_metrics.py` - Regression metrics
3. `03_cross_validation.py` - Validation techniques
4. `exercises.py` - Practice problems
