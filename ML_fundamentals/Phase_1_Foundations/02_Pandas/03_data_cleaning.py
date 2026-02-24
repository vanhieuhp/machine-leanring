"""
Pandas Fundamentals - Part 3: Data Cleaning
=============================================

This module covers:
- Handling missing values
- Removing duplicates
- Data type conversion
- Outlier detection
- Data normalization
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. HANDLING MISSING VALUES
# ============================================================================

print("=" * 70)
print("1. HANDLING MISSING VALUES")
print("=" * 70)

# Create data with missing values
data = {
    'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'age': [25, None, 35, 28, 32],
    'salary': [50000, 60000, 75000, None, 65000],
    'department': ['Sales', 'IT', 'HR', 'Sales', None]
}

df = pd.DataFrame(data)
print("DataFrame with missing values:")
print(df)

print("\n\nMissing values count:")
print(df.isnull().sum())

print("\n\nMissing values percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

# ============================================================================
# 2. REMOVING MISSING VALUES
# ============================================================================

print("\n" + "=" * 70)
print("2. REMOVING MISSING VALUES")
print("=" * 70)

# Drop rows with any missing values
df_dropped = df.dropna()
print("After dropna():")
print(df_dropped)

# Drop rows where specific column is missing
df_dropped_age = df.dropna(subset=['age'])
print("\n\nAfter dropna(subset=['age']):")
print(df_dropped_age)

# Drop columns with missing values
df_dropped_cols = df.dropna(axis=1)
print("\n\nAfter dropna(axis=1):")
print(df_dropped_cols)

# ============================================================================
# 3. FILLING MISSING VALUES
# ============================================================================

print("\n" + "=" * 70)
print("3. FILLING MISSING VALUES")
print("=" * 70)

df = pd.DataFrame(data)

# Fill with constant value
df_filled_const = df.fillna('Unknown')
print("After fillna('Unknown'):")
print(df_filled_const)

# Fill with mean (for numerical columns)
df_filled_mean = df.copy()
df_filled_mean['age'] = df_filled_mean['age'].fillna(df_filled_mean['age'].mean())
df_filled_mean['salary'] = df_filled_mean['salary'].fillna(df_filled_mean['salary'].mean())
print("\n\nAfter filling with mean:")
print(df_filled_mean)

# Forward fill
df_ffill = df.fillna(method='ffill')
print("\n\nAfter forward fill:")
print(df_ffill)

# Backward fill
df_bfill = df.fillna(method='bfill')
print("\n\nAfter backward fill:")
print(df_bfill)

# ============================================================================
# 4. REMOVING DUPLICATES
# ============================================================================

print("\n" + "=" * 70)
print("4. REMOVING DUPLICATES")
print("=" * 70)

data_dup = {
    'name': ['Alice', 'Bob', 'Alice', 'David', 'Bob'],
    'age': [25, 30, 25, 28, 30],
    'salary': [50000, 60000, 50000, 55000, 60000]
}

df_dup = pd.DataFrame(data_dup)
print("DataFrame with duplicates:")
print(df_dup)

print("\n\nDuplicate rows:")
print(df_dup.duplicated())

# Remove duplicates
df_no_dup = df_dup.drop_duplicates()
print("\n\nAfter drop_duplicates():")
print(df_no_dup)

# Remove duplicates based on specific column
df_no_dup_name = df_dup.drop_duplicates(subset=['name'])
print("\n\nAfter drop_duplicates(subset=['name']):")
print(df_no_dup_name)

# ============================================================================
# 5. DATA TYPE CONVERSION
# ============================================================================

print("\n" + "=" * 70)
print("5. DATA TYPE CONVERSION")
print("=" * 70)

data_types = {
    'age': ['25', '30', '35', '28'],
    'salary': ['50000', '60000', '75000', '55000'],
    'is_manager': ['True', 'False', 'True', 'False']
}

df_types = pd.DataFrame(data_types)
print("Original DataFrame:")
print(df_types)
print(f"\nData types:\n{df_types.dtypes}")

# Convert to numeric
df_types['age'] = pd.to_numeric(df_types['age'])
df_types['salary'] = pd.to_numeric(df_types['salary'])
print("\n\nAfter converting to numeric:")
print(df_types)
print(f"\nData types:\n{df_types.dtypes}")

# Convert to boolean
df_types['is_manager'] = df_types['is_manager'].astype(bool)
print("\n\nAfter converting to boolean:")
print(df_types)
print(f"\nData types:\n{df_types.dtypes}")

# ============================================================================
# 6. OUTLIER DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("6. OUTLIER DETECTION")
print("=" * 70)

# Create data with outliers
salaries = [50000, 55000, 60000, 65000, 70000, 75000, 500000]  # 500000 is outlier
df_outliers = pd.DataFrame({'salary': salaries})

print("Salaries with outlier:")
print(df_outliers)

# Using IQR method
Q1 = df_outliers['salary'].quantile(0.25)
Q3 = df_outliers['salary'].quantile(0.75)
IQR = Q3 - Q1

print(f"\nQ1: {Q1}, Q3: {Q3}, IQR: {IQR}")

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

outliers = df_outliers[(df_outliers['salary'] < lower_bound) | (df_outliers['salary'] > upper_bound)]
print(f"\nOutliers:\n{outliers}")

# Remove outliers
df_no_outliers = df_outliers[(df_outliers['salary'] >= lower_bound) & (df_outliers['salary'] <= upper_bound)]
print(f"\nAfter removing outliers:\n{df_no_outliers}")

# ============================================================================
# 7. STRING OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("7. STRING OPERATIONS")
print("=" * 70)

data_str = {
    'name': ['alice smith', 'bob jones', 'charlie brown'],
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
}

df_str = pd.DataFrame(data_str)
print("Original DataFrame:")
print(df_str)

# Convert to uppercase
df_str['name_upper'] = df_str['name'].str.upper()
print("\n\nAfter upper():")
print(df_str)

# Extract domain from email
df_str['domain'] = df_str['email'].str.split('@').str[1]
print("\n\nAfter extracting domain:")
print(df_str)

# Check if contains
df_str['is_gmail'] = df_str['email'].str.contains('gmail')
print("\n\nAfter checking for gmail:")
print(df_str)

# ============================================================================
# 8. NORMALIZATION AND SCALING
# ============================================================================

print("\n" + "=" * 70)
print("8. NORMALIZATION AND SCALING")
print("=" * 70)

data_scale = {
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 65000]
}

df_scale = pd.DataFrame(data_scale)
print("Original DataFrame:")
print(df_scale)

# Min-Max normalization (0-1)
df_scale['age_normalized'] = (df_scale['age'] - df_scale['age'].min()) / (df_scale['age'].max() - df_scale['age'].min())
df_scale['salary_normalized'] = (df_scale['salary'] - df_scale['salary'].min()) / (df_scale['salary'].max() - df_scale['salary'].min())

print("\n\nAfter Min-Max normalization:")
print(df_scale)

# Standardization (z-score)
df_scale['age_standardized'] = (df_scale['age'] - df_scale['age'].mean()) / df_scale['age'].std()
df_scale['salary_standardized'] = (df_scale['salary'] - df_scale['salary'].mean()) / df_scale['salary'].std()

print("\n\nAfter standardization:")
print(df_scale)

# ============================================================================
# 9. PRACTICAL EXAMPLE: Data Cleaning Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL EXAMPLE: Data Cleaning Pipeline")
print("=" * 70)

# Create messy data
messy_data = {
    'id': [1, 2, 3, 3, 5, 6, 7],
    'name': ['alice', 'bob', 'charlie', 'charlie', 'eve', 'frank', None],
    'age': ['25', '30', '35', '35', '28', '999', '32'],
    'salary': [50000, 60000, 75000, 75000, None, 55000, 65000],
    'department': ['Sales', 'IT', 'HR', 'HR', 'Sales', 'IT', 'Finance']
}

df_messy = pd.DataFrame(messy_data)
print("Messy DataFrame:")
print(df_messy)

# Cleaning steps
df_clean = df_messy.copy()

# 1. Remove duplicates
df_clean = df_clean.drop_duplicates()
print("\n\n1. After removing duplicates:")
print(df_clean)

# 2. Remove rows with missing names
df_clean = df_clean.dropna(subset=['name'])
print("\n\n2. After removing missing names:")
print(df_clean)

# 3. Convert age to numeric and remove outliers
df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
df_clean = df_clean[df_clean['age'] < 100]
print("\n\n3. After cleaning age:")
print(df_clean)

# 4. Fill missing salary with mean
df_clean['salary'] = df_clean['salary'].fillna(df_clean['salary'].mean())
print("\n\n4. After filling missing salary:")
print(df_clean)

# 5. Standardize names
df_clean['name'] = df_clean['name'].str.title()
print("\n\n5. After standardizing names:")
print(df_clean)

print("\n\nFinal cleaned DataFrame:")
print(df_clean)
