# Statistics Fundamentals Guide

## What is Statistics?

Statistics is the science of collecting, analyzing, and interpreting data. It provides:
- **Descriptive statistics** - summarize data (mean, median, std)
- **Probability** - likelihood of events
- **Distributions** - patterns in data
- **Hypothesis testing** - make decisions from data
- **Correlation** - relationships between variables

## Why Statistics Matters for ML

ML is fundamentally about statistics:
- Understand data distributions
- Detect outliers and anomalies
- Measure relationships between features
- Evaluate model performance
- Make predictions with uncertainty

## Learning Objectives

By the end of this section, you'll understand:
1. Descriptive statistics (mean, median, std, etc.)
2. Probability distributions
3. Correlation and covariance
4. Z-scores and standardization
5. Basic hypothesis testing

## Key Concepts

### 1. Descriptive Statistics

- **Mean** - average value
- **Median** - middle value
- **Mode** - most frequent value
- **Std Dev** - spread of data
- **Variance** - squared spread
- **Range** - max - min

### 2. Distributions

- **Normal** - bell curve, symmetric
- **Uniform** - all values equally likely
- **Binomial** - yes/no outcomes
- **Poisson** - count of events

### 3. Correlation

- **Pearson** - linear relationship (-1 to 1)
- **Spearman** - rank-based relationship
- **Covariance** - joint variability

### 4. Z-Score

Standardized score showing how many standard deviations from mean:
```
z = (x - mean) / std
```

## Study Path

1. **Start with**: `01_descriptive_stats.py`
   - Calculate basic statistics
   - Understand distributions
   - Explore data

2. **Then**: `02_distributions.py`
   - Normal distribution
   - Other distributions
   - Probability calculations

3. **Practice**: `exercises.py`
   - Apply statistics to real data
   - Build intuition

## Common Mistakes to Avoid

1. **Confusing mean and median**
   - Mean affected by outliers
   - Median more robust

2. **Ignoring distribution shape**
   - Normal distribution assumptions
   - Skewed data needs different approaches

3. **Misinterpreting correlation**
   - Correlation ≠ causation
   - Check scatter plots

4. **Wrong statistical test**
   - Different tests for different data types
   - Check assumptions first

5. **Ignoring sample size**
   - Small samples have high variability
   - Larger samples more reliable

## Tips for Learning

- Always visualize distributions
- Calculate statistics by hand first
- Understand what each statistic means
- Check assumptions before tests
- Use domain knowledge to interpret results

## Next Steps

After mastering Statistics:
- Use in feature engineering
- Evaluate model performance
- Make predictions with confidence intervals
- Understand ML algorithms mathematically
