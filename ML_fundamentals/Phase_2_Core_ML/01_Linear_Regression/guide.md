# Linear Regression Guide

## What is Linear Regression?

Linear regression finds the best-fitting line through data points. It's used to predict continuous values based on one or more features.

## When to Use

- House prices based on size
- Temperature based on time of day
- Stock prices based on historical data
- Salary based on experience

## Key Concepts

### 1. The Linear Model
```
y = mx + b
```
- `y`: predicted value
- `x`: input feature
- `m`: slope (how much y changes per unit x)
- `b`: intercept (y value when x=0)

### 2. Multiple Features
```
y = m₁x₁ + m₂x₂ + ... + mₙxₙ + b
```
- Extend to multiple input features
- Each feature has its own coefficient

### 3. Cost Function (Loss)
```
MSE = (1/n) * Σ(y_pred - y_actual)²
```
- Measures how far predictions are from actual values
- Goal: minimize MSE

### 4. Gradient Descent
Algorithm to find the best coefficients:
1. Start with random coefficients
2. Calculate gradient (direction of steepest descent)
3. Update coefficients in that direction
4. Repeat until convergence

### 5. Evaluation Metrics

**R² Score** (coefficient of determination):
- Ranges from 0 to 1
- 1 = perfect fit
- 0 = model no better than mean

**RMSE** (Root Mean Squared Error):
- Same units as target variable
- Lower is better

**MAE** (Mean Absolute Error):
- Average absolute error
- More robust to outliers than RMSE

## Implementation Steps

1. **Prepare data**: Load, clean, split
2. **Scale features**: Normalize to same range
3. **Create model**: Initialize coefficients
4. **Train**: Use gradient descent
5. **Evaluate**: Check R², RMSE, MAE
6. **Predict**: Make predictions on new data

## Assumptions

Linear regression assumes:
1. Linear relationship between features and target
2. Independent observations
3. Normally distributed errors
4. Constant variance (homoscedasticity)

## Advantages

- Simple and interpretable
- Fast to train
- Works well with linear relationships
- Good baseline model

## Disadvantages

- Assumes linear relationship
- Sensitive to outliers
- Requires feature scaling
- Can underfit complex data

## Study Files

1. `01_simple_regression.py` - Single feature
2. `02_multiple_regression.py` - Multiple features
3. `03_gradient_descent.py` - Algorithm details
4. `exercises.py` - Practice problems

## Next: Logistic Regression

After mastering linear regression, learn logistic regression for classification problems.
