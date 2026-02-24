"""
Pandas Fundamentals - Part 1: DataFrames Basics
================================================

This module covers:
- Creating DataFrames
- DataFrame properties and structure
- Indexing and selection
- Basic operations
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. CREATING DATAFRAMES
# ============================================================================

print("=" * 70)
print("1. CREATING DATAFRAMES")
print("=" * 70)

# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000],
    'department': ['Sales', 'IT', 'HR', 'Sales']
}

df = pd.DataFrame(data)
print("DataFrame from dictionary:")
print(df)

# From list of lists
data_list = [
    ['Alice', 25, 50000],
    ['Bob', 30, 60000],
    ['Charlie', 35, 75000]
]

df2 = pd.DataFrame(data_list, columns=['name', 'age', 'salary'])
print("\n\nDataFrame from list of lists:")
print(df2)

# From NumPy array
arr = np.random.rand(3, 4)
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])
print("\n\nDataFrame from NumPy array:")
print(df3)

# ============================================================================
# 2. DATAFRAME PROPERTIES
# ============================================================================

print("\n" + "=" * 70)
print("2. DATAFRAME PROPERTIES")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000]
})

print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Size: {df.size}")    # total elements
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print(f"Data types:\n{df.dtypes}")

# ============================================================================
# 3. VIEWING DATA
# ============================================================================

print("\n" + "=" * 70)
print("3. VIEWING DATA")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 65000]
})

print("First 2 rows:")
print(df.head(2))

print("\n\nLast 2 rows:")
print(df.tail(2))

print("\n\nInfo:")
print(df.info())

print("\n\nStatistical summary:")
print(df.describe())

# ============================================================================
# 4. INDEXING - LABEL-BASED (loc)
# ============================================================================

print("\n" + "=" * 70)
print("4. INDEXING - LABEL-BASED (loc)")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 75000]
})

print("Original DataFrame:")
print(df)

# Get single column
print(f"\nColumn 'name':\n{df['name']}")

# Get single row
print(f"\nRow 0:\n{df.loc[0]}")

# Get specific cell
print(f"\nCell [0, 'name']: {df.loc[0, 'name']}")

# Get multiple rows
print(f"\nRows 0-1:\n{df.loc[0:1]}")

# Get multiple columns
print(f"\nColumns 'name' and 'age':\n{df[['name', 'age']]}")

# ============================================================================
# 5. INDEXING - POSITION-BASED (iloc)
# ============================================================================

print("\n" + "=" * 70)
print("5. INDEXING - POSITION-BASED (iloc)")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 75000]
})

# Get first row
print(f"First row (iloc[0]):\n{df.iloc[0]}")

# Get first column
print(f"\nFirst column (iloc[:, 0]):\n{df.iloc[:, 0]}")

# Get specific cell
print(f"\nCell [0, 1]: {df.iloc[0, 1]}")

# Get submatrix
print(f"\nRows 0-1, columns 0-1:\n{df.iloc[0:2, 0:2]}")

# ============================================================================
# 6. FILTERING
# ============================================================================

print("\n" + "=" * 70)
print("6. FILTERING")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 65000]
})

print("Original DataFrame:")
print(df)

# Filter by condition
print("\n\nAge > 28:")
print(df[df['age'] > 28])

# Multiple conditions
print("\n\nAge > 28 AND salary > 60000:")
print(df[(df['age'] > 28) & (df['salary'] > 60000)])

# Filter by value
print("\n\nName == 'Alice':")
print(df[df['name'] == 'Alice'])

# ============================================================================
# 7. ADDING AND REMOVING COLUMNS
# ============================================================================

print("\n" + "=" * 70)
print("7. ADDING AND REMOVING COLUMNS")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 75000]
})

print("Original DataFrame:")
print(df)

# Add new column
df['bonus'] = df['salary'] * 0.1
print("\n\nAfter adding 'bonus' column:")
print(df)

# Add column with same value
df['department'] = 'Sales'
print("\n\nAfter adding 'department' column:")
print(df)

# Remove column
df_dropped = df.drop('bonus', axis=1)
print("\n\nAfter dropping 'bonus' column:")
print(df_dropped)

# ============================================================================
# 8. SORTING
# ============================================================================

print("\n" + "=" * 70)
print("8. SORTING")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000]
})

print("Original DataFrame:")
print(df)

# Sort by column
print("\n\nSorted by age:")
print(df.sort_values('age'))

# Sort by multiple columns
print("\n\nSorted by age (descending):")
print(df.sort_values('age', ascending=False))

# Sort by index
print("\n\nSorted by index (descending):")
print(df.sort_index(ascending=False))

# ============================================================================
# 9. RENAMING COLUMNS
# ============================================================================

print("\n" + "=" * 70)
print("9. RENAMING COLUMNS")
print("=" * 70)

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 75000]
})

print("Original DataFrame:")
print(df)

# Rename specific columns
df_renamed = df.rename(columns={'name': 'employee_name', 'salary': 'annual_salary'})
print("\n\nAfter renaming:")
print(df_renamed)

# ============================================================================
# 10. PRACTICAL EXAMPLE: Employee Data
# ============================================================================

print("\n" + "=" * 70)
print("10. PRACTICAL EXAMPLE: Employee Data")
print("=" * 70)

employees = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 65000],
    'department': ['Sales', 'IT', 'HR', 'Sales', 'IT']
})

print("Employee DataFrame:")
print(employees)

# Get IT department employees
print("\n\nIT Department:")
print(employees[employees['department'] == 'IT'])

# Get employees with salary > 60000
print("\n\nHigh earners (> 60000):")
print(employees[employees['salary'] > 60000])

# Add bonus column
employees['bonus'] = employees['salary'] * 0.1
print("\n\nWith bonus:")
print(employees)

# Get average salary by department
print("\n\nAverage salary by department:")
print(employees.groupby('department')['salary'].mean())
