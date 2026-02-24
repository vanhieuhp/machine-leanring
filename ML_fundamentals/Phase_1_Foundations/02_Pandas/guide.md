# Pandas Fundamentals Guide

## What is Pandas?

Pandas is the primary data manipulation library in Python. It provides:
- **DataFrames** - 2D labeled data structures (like Excel spreadsheets)
- **Series** - 1D labeled arrays
- **Data cleaning** - handle missing values, duplicates
- **Data transformation** - reshape, pivot, merge data
- **File I/O** - read/write CSV, Excel, SQL, JSON

## Why Pandas Matters for ML

80% of ML work is data preparation. Pandas is essential for:
- Loading and exploring datasets
- Cleaning messy real-world data
- Feature engineering
- Handling missing values
- Preparing data for ML algorithms

## Learning Objectives

By the end of this section, you'll understand:
1. Creating and manipulating DataFrames
2. Loading data from files
3. Data cleaning and preprocessing
4. Grouping and aggregation
5. Merging and joining data

## Key Concepts

### 1. Series vs DataFrame

**Series** - 1D labeled array:
```python
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
```

**DataFrame** - 2D labeled table:
```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})
```

### 2. Indexing

- **Label-based**: `df.loc['row_label', 'column_name']`
- **Position-based**: `df.iloc[0, 1]`
- **Column access**: `df['column_name']` or `df.column_name`

### 3. Data Types

Pandas automatically infers types:
- `int64` - integers
- `float64` - decimals
- `object` - strings, mixed types
- `datetime64` - dates and times
- `bool` - True/False

## Study Path

1. **Start with**: `01_dataframes_basics.py`
   - Create DataFrames
   - Understand structure and properties
   - Basic indexing

2. **Then**: `02_data_loading.py`
   - Load CSV files
   - Explore data
   - Handle different file formats

3. **Next**: `03_data_cleaning.py`
   - Handle missing values
   - Remove duplicates
   - Data type conversion
   - Outlier detection

4. **Practice**: `exercises.py`
   - Apply techniques to real data
   - Build data cleaning skills

## Common Mistakes to Avoid

1. **Not checking data types**
   - Use `df.info()` and `df.dtypes`
   - Convert types explicitly when needed

2. **Ignoring missing values**
   - Check with `df.isnull().sum()`
   - Handle appropriately (drop, fill, or impute)

3. **Modifying original data**
   - Use `.copy()` when needed
   - Be aware of view vs copy behavior

4. **Not exploring before cleaning**
   - Always use `df.head()`, `df.describe()`, `df.info()`
   - Understand data before making changes

5. **Inefficient operations**
   - Use vectorized operations, not loops
   - Use `.apply()` or `.map()` for custom functions

## Tips for Learning

- Always start with `df.head()` and `df.info()`
- Use `df.describe()` for numerical summaries
- Check `df.isnull().sum()` for missing values
- Print shapes and dtypes frequently
- Use `df.value_counts()` to understand categorical data

## Next Steps

After mastering Pandas:
- Combine with NumPy for numerical operations
- Use with Matplotlib for visualization
- Prepare data for ML algorithms
- Build complete data pipelines
