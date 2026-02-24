# NumPy Fundamentals Guide

## What is NumPy?

NumPy is the foundation of numerical computing in Python. It provides:
- **N-dimensional arrays** (ndarrays) - efficient storage and computation
- **Mathematical functions** - linear algebra, statistics, random numbers
- **Broadcasting** - operations on arrays of different shapes
- **Performance** - 10-100x faster than Python lists

## Why NumPy Matters for ML

All machine learning algorithms work with numerical arrays. NumPy is the backbone:
- Store features and targets as arrays
- Perform matrix operations for model training
- Efficient computation on large datasets

## Learning Objectives

By the end of this section, you'll understand:
1. Creating and manipulating arrays
2. Array operations and broadcasting
3. Matrix operations
4. Statistical functions
5. Random number generation

## Key Concepts

### 1. Arrays vs Lists

**Python List:**
```python
my_list = [1, 2, 3, 4, 5]
# Slow, uses more memory, mixed types allowed
```

**NumPy Array:**
```python
import numpy as np
my_array = np.array([1, 2, 3, 4, 5])
# Fast, memory efficient, homogeneous types
```

### 2. Array Shapes

Understanding shapes is crucial:
- **1D array**: `(5,)` - single row
- **2D array**: `(3, 4)` - 3 rows, 4 columns
- **3D array**: `(2, 3, 4)` - 2 matrices of 3x4

### 3. Broadcasting

NumPy automatically aligns arrays of different shapes:
```python
array = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
scalar = 10
result = array + scalar  # Broadcasts scalar to all elements
```

## Study Path

1. **Start with**: `01_arrays_basics.py`
   - Learn array creation and properties
   - Understand indexing and slicing

2. **Then**: `02_operations.py`
   - Element-wise operations
   - Broadcasting rules
   - Aggregation functions

3. **Next**: `03_matrix_operations.py`
   - Dot products
   - Matrix multiplication
   - Transpose and reshape

4. **Practice**: `exercises.py`
   - Apply what you learned
   - Build intuition

## Common Mistakes to Avoid

1. **Confusing shape and size**
   - Shape: dimensions (e.g., 3x4)
   - Size: total elements (e.g., 12)

2. **Forgetting broadcasting rules**
   - Arrays must be compatible in dimensions
   - Dimensions are compared from right to left

3. **Modifying original arrays**
   - Use `.copy()` if you need to preserve original
   - Slicing creates views, not copies

4. **Type mismatches**
   - NumPy arrays are homogeneous
   - Mixed types get converted to common type

## Tips for Learning

- Run each example and modify it
- Print shapes and types to understand data
- Use `array.shape`, `array.dtype`, `array.ndim` frequently
- Visualize operations mentally before running

## Next Steps

After mastering NumPy:
- Move to Pandas for data manipulation
- Use NumPy arrays as input to ML algorithms
- Combine with Matplotlib for visualization
