"""
Pandas Fundamentals - Part 2: Data Loading and Exploration
===========================================================

This module covers:
- Loading data from CSV files
- Exploring data structure
- Understanding data types
- Basic data exploration techniques
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. LOADING DATA FROM CSV
# ============================================================================

print("=" * 70)
print("1. LOADING DATA FROM CSV")
print("=" * 70)

# Create sample CSV file for demonstration
sample_data = """name,age,salary,department,hire_date
Alice,25,50000,Sales,2020-01-15
Bob,30,60000,IT,2019-03-22
Charlie,35,75000,HR,2018-06-10
David,28,55000,Sales,2021-02-14
Eve,32,65000,IT,2020-11-30"""

with open('sample_employees.csv', 'w') as f:
    f.write(sample_data)

# Load CSV file
df = pd.read_csv('sample_employees.csv')
print("Loaded DataFrame:")
print(df)

# ============================================================================
# 2. BASIC EXPLORATION
# ============================================================================

print("\n" + "=" * 70)
print("2. BASIC EXPLORATION")
print("=" * 70)

print(f"Shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst few rows:")
print(df.head(3))
print(f"\nLast few rows:")
print(df.tail(2))

# ============================================================================
# 3. DATA INFO
# ============================================================================

print("\n" + "=" * 70)
print("3. DATA INFO")
print("=" * 70)

print("DataFrame info:")
df.info()

# ============================================================================
# 4. STATISTICAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("4. STATISTICAL SUMMARY")
print("=" * 70)

print("Numerical columns summary:")
print(df.describe())

print("\n\nAll columns summary:")
print(df.describe(include='all'))

# ============================================================================
# 5. CHECKING FOR MISSING VALUES
# ============================================================================

print("\n" + "=" * 70)
print("5. CHECKING FOR MISSING VALUES")
print("=" * 70)

# Create data with missing values
data_with_missing = """name,age,salary,department
Alice,25,50000,Sales
Bob,,60000,IT
Charlie,35,,HR
David,28,55000,
Eve,32,65000,IT"""

with open('sample_missing.csv', 'w') as f:
    f.write(data_with_missing)

df_missing = pd.read_csv('sample_missing.csv')
print("DataFrame with missing values:")
print(df_missing)

print("\n\nMissing values count:")
print(df_missing.isnull().sum())

print("\n\nMissing values percentage:")
print((df_missing.isnull().sum() / len(df_missing) * 100).round(2))

print("\n\nRows with any missing values:")
print(df_missing[df_missing.isnull().any(axis=1)])

# ============================================================================
# 6. UNIQUE VALUES AND VALUE COUNTS
# ============================================================================

print("\n" + "=" * 70)
print("6. UNIQUE VALUES AND VALUE COUNTS")
print("=" * 70)

df = pd.read_csv('sample_employees.csv')

print("Unique departments:")
print(df['department'].unique())

print("\n\nNumber of unique departments:")
print(df['department'].nunique())

print("\n\nValue counts for department:")
print(df['department'].value_counts())

# ============================================================================
# 7. GROUPBY OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("7. GROUPBY OPERATIONS")
print("=" * 70)

print("Average salary by department:")
print(df.groupby('department')['salary'].mean())

print("\n\nCount of employees by department:")
print(df.groupby('department').size())

print("\n\nMultiple aggregations:")
print(df.groupby('department')['salary'].agg(['mean', 'min', 'max', 'count']))

# ============================================================================
# 8. SORTING AND RANKING
# ============================================================================

print("\n" + "=" * 70)
print("8. SORTING AND RANKING")
print("=" * 70)

print("Sorted by salary (ascending):")
print(df.sort_values('salary'))

print("\n\nSorted by salary (descending):")
print(df.sort_values('salary', ascending=False))

print("\n\nSorted by multiple columns:")
print(df.sort_values(['department', 'salary']))

# ============================================================================
# 9. FILTERING AND SELECTION
# ============================================================================

print("\n" + "=" * 70)
print("9. FILTERING AND SELECTION")
print("=" * 70)

print("Employees with salary > 60000:")
print(df[df['salary'] > 60000])

print("\n\nIT department employees:")
print(df[df['department'] == 'IT'])

print("\n\nEmployees aged 25-30:")
print(df[(df['age'] >= 25) & (df['age'] <= 30)])

# ============================================================================
# 10. CREATING NEW COLUMNS
# ============================================================================

print("\n" + "=" * 70)
print("10. CREATING NEW COLUMNS")
print("=" * 70)

df_copy = df.copy()

# Add bonus column
df_copy['bonus'] = df_copy['salary'] * 0.1
print("With bonus column:")
print(df_copy)

# Add salary category
df_copy['salary_category'] = pd.cut(df_copy['salary'],
                                     bins=[0, 55000, 65000, 100000],
                                     labels=['Low', 'Medium', 'High'])
print("\n\nWith salary category:")
print(df_copy)

# ============================================================================
# 11. PRACTICAL EXAMPLE: Sales Data Analysis
# ============================================================================

print("\n" + "=" * 70)
print("11. PRACTICAL EXAMPLE: Sales Data Analysis")
print("=" * 70)

# Create sales data
sales_data = """date,product,quantity,price,region
2024-01-01,Laptop,2,1000,North
2024-01-02,Mouse,5,25,South
2024-01-03,Keyboard,3,75,North
2024-01-04,Laptop,1,1000,East
2024-01-05,Monitor,2,300,South
2024-01-06,Mouse,10,25,North
2024-01-07,Keyboard,4,75,East
2024-01-08,Laptop,1,1000,South"""

with open('sales_data.csv', 'w') as f:
    f.write(sales_data)

sales_df = pd.read_csv('sales_data.csv')
print("Sales DataFrame:")
print(sales_df)

# Calculate total revenue
sales_df['revenue'] = sales_df['quantity'] * sales_df['price']
print("\n\nWith revenue column:")
print(sales_df)

# Revenue by product
print("\n\nRevenue by product:")
print(sales_df.groupby('product')['revenue'].sum().sort_values(ascending=False))

# Revenue by region
print("\n\nRevenue by region:")
print(sales_df.groupby('region')['revenue'].sum())

# Top selling product
print("\n\nTop selling product by quantity:")
print(sales_df.groupby('product')['quantity'].sum().sort_values(ascending=False))

# Clean up
import os
os.remove('sample_employees.csv')
os.remove('sample_missing.csv')
os.remove('sales_data.csv')
