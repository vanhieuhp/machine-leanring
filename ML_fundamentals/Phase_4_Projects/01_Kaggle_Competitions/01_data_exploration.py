"""
Kaggle Competitions - Part 1: Data Exploration
==============================================

This module covers:
- Loading competition data
- Understanding data structure
- Identifying data types
- Handling missing values
- Visualizing distributions
- Analyzing correlations

Based on: Titanic Competition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# 1. LOADING DATA
# ============================================================================

print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

# For Kaggle competitions, data is typically in CSV format
# We'll create sample Titanic-like data for demonstration

# Sample Titanic data (in practice, load from Kaggle)
train_data = {
    'PassengerId': range(1, 891),
    'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] + [0] * 880,
    'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 1, 1] + list(np.random.choice([1, 2, 3], 880)),
    'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina',
             'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry', 'Moran, Mr. James',
             'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard', 'Johnson, Mrs. Oscar W',
             'Nasser, Mrs. Nicholas'] + [f'Passenger_{i}' for i in range(11, 891)],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female']
            + list(np.random.choice(['male', 'female'], 880)),
    'Age': [22.0, 38.0, 26.0, 35.0, 35.0, np.nan, 54.0, 2.0, 27.0, 14.0]
           + list(np.random.uniform(1, 70, 880)),
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1] + list(np.random.randint(0, 5, 880)),
    'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] + list(np.random.randint(0, 6, 880)),
    'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450',
               '330877', '17463', '349909', '347742', '237736']
              + [f'TICKET_{i}' for i in range(11, 891)],
    'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708]
             + list(np.random.uniform(0, 300, 880)),
    'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan, np.nan, 'D6']
             + [np.nan] * 880,
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'C', 'S']
                + list(np.random.choice(['S', 'C', 'Q'], 870, p=[0.55, 0.25, 0.20]))
}

df = pd.DataFrame(train_data)

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# ============================================================================
# 2. DATA STRUCTURE
# ============================================================================

print("\n" + "=" * 70)
print("2. DATA STRUCTURE")
print("=" * 70)

# Column names and data types
print("\nColumn names and data types:")
print(df.dtypes)

print("\n" + "-" * 50)
print("First 5 rows:")
print(df.head())

print("\n" + "-" * 50)
print("Last 5 rows:")
print(df.tail())

# ============================================================================
# 3. DATA TYPES CATEGORIZATION
# ============================================================================

print("\n" + "=" * 70)
print("3. DATA TYPES CATEGORIZATION")
print("=" * 70)

# Identify different types of columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}):")
print(numerical_cols)

print(f"\nCategorical columns ({len(categorical_cols)}):")
print(categorical_cols)

# Identify target column
target_col = 'Survived'
print(f"\nTarget column: {target_col}")
print(f"Target distribution:\n{df[target_col].value_counts()}")
print(f"Survival rate: {df[target_col].mean()*100:.2f}%")

# ============================================================================
# 4. MISSING VALUES
# ============================================================================

print("\n" + "=" * 70)
print("4. MISSING VALUES ANALYSIS")
print("=" * 70)

# Count missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

print("\nMissing values summary:")
print(missing_df[missing_df['Missing Count'] > 0])

# Visualize missing values
print("\nMissing values visualization:")
for col in df.columns:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        pct = (missing_count / len(df)) * 100
        print(f"  {col}: {'█' * int(pct/2)} {pct:.1f}%")

# ============================================================================
# 5. STATISTICAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("5. STATISTICAL SUMMARY")
print("=" * 70)

# Numerical columns statistics
print("\nNumerical statistics:")
print(df.describe())

print("\n" + "-" * 50)
print("\nCategorical columns statistics:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    if df[col].nunique() < 10:
        print(f"  Value counts:\n{df[col].value_counts()}")

# ============================================================================
# 6. DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("6. DISTRIBUTION ANALYSIS")
print("=" * 70)

# Analyze distributions
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']

for col in numerical_features:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std: {df[col].std():.2f}")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")
    print(f"  Skewness: {df[col].skew():.2f}")

# Analyze categorical distributions
categorical_features = ['Pclass', 'Sex', 'Embarked']

for col in categorical_features:
    print(f"\n{col} distribution:")
    dist = df[col].value_counts(normalize=True) * 100
    for val, pct in dist.items():
        print(f"  {val}: {pct:.1f}%")

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("7. CORRELATION ANALYSIS")
print("=" * 70)

# Calculate correlations with target
correlations = df[numerical_cols].corr()[target_col].sort_values(ascending=False)
print(f"\nCorrelations with {target_col}:")
print(correlations)

# Full correlation matrix
print("\nFull correlation matrix (numerical columns):")
corr_matrix = df[numerical_cols].corr()
print(corr_matrix)

# ============================================================================
# 8. FEATURE RELATIONSHIP WITH TARGET
# ============================================================================

print("\n" + "=" * 70)
print("8. FEATURE RELATIONSHIP WITH TARGET")
print("=" * 70)

# Survival rate by categorical features
print("\nSurvival rate by Pclass:")
print(df.groupby('Pclass')['Survived'].mean())

print("\nSurvival rate by Sex:")
print(df.groupby('Sex')['Survived'].mean())

print("\nSurvival rate by Embarked:")
print(df.groupby('Embarked')['Survived'].mean())

# Survival rate by numerical features (binned)
print("\nSurvival rate by Age groups:")
df['Age_group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])
print(df.groupby('Age_group')['Survived'].mean())

print("\nSurvival rate by Fare quartiles:")
df['Fare_quartile'] = pd.qcut(df['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(df.groupby('Fare_quartile')['Survived'].mean())

# ============================================================================
# 9. OUTLIER DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("9. OUTLIER DETECTION")
print("=" * 70)

# Using IQR method
for col in ['Age', 'Fare', 'SibSp', 'Parch']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

    print(f"\n{col}:")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# ============================================================================
# 10. PRACTICAL EXAMPLE: COMPREHENSIVE EDA REPORT
# ============================================================================

print("\n" + "=" * 70)
print("10. COMPREHENSIVE EDA REPORT")
print("=" * 70)

def generate_eda_report(df, target_col):
    """Generate a comprehensive EDA report."""

    report = {
        'Dataset Overview': {
            'Shape': df.shape,
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        },
        'Target Variable': {
            'Column': target_col,
            'Type': df[target_col].dtype,
            'Distribution': df[target_col].value_counts().to_dict()
        },
        'Missing Values': {
            'Total': df.isnull().sum().sum(),
            'Columns with Missing': df.columns[df.isnull().any()].tolist()
        },
        'Numerical Features': {
            'Count': len(df.select_dtypes(include=['int64', 'float64']).columns),
            'Columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        },
        'Categorical Features': {
            'Count': len(df.select_dtypes(include=['object']).columns),
            'Columns': df.select_dtypes(include=['object']).columns.tolist()
        }
    }

    return report

report = generate_eda_report(df, target_col)

for section, details in report.items():
    print(f"\n{section}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. ALWAYS start with data exploration
   - Understand the structure before modeling
   - Identify data types, missing values, distributions

2. Missing values matter
   - Check which columns have missing values
   - Decide on imputation strategy (mean, median, mode, KNN, etc.)

3. Understand your target
   - Binary classification: check class balance
   - Regression: check target distribution and skewness

4. Feature relationships
   - Correlation analysis for numerical features
   - Group statistics for categorical features

5. Outliers exist
   - Identify and decide how to handle them
   - Can indicate data quality issues or important patterns

6. Visualize when possible
   - Distributions, boxplots, scatter plots
   - Seaborn for quick statistical visualizations

Next: Feature Engineering (02_feature_engineering.py)
""")
