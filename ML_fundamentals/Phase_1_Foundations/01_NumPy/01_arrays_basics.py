"""
NumPy Fundamentals - Part 1: Arrays Basics
==========================================

This module covers:
- Creating arrays
- Array properties (shape, dtype, size)
- Indexing and slicing
- Array types and dtypes
"""

import numpy as np

# ============================================================================
# 1. CREATING ARRAYS
# ============================================================================

print("=" * 70)
print("1. CREATING ARRAYS")
print("=" * 70)

# From Python list
arr1 = np.array([1, 2, 3, 4, 5])
print(f"From list: {arr1}")
print(f"Type: {type(arr1)}")

# 2D array (matrix)
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D array:\n{arr2d}")

# Using built-in functions
zeros = np.zeros((3, 4))  # 3x4 matrix of zeros
print(f"\nZeros (3x4):\n{zeros}")

ones = np.ones((2, 3))  # 2x3 matrix of ones
print(f"\nOnes (2x3):\n{ones}")

# Range of values
range_arr = np.arange(0, 10, 2)  # Start, stop, step
print(f"\nRange (0 to 10, step 2): {range_arr}")

# Evenly spaced values
linspace_arr = np.linspace(0, 1, 5)  # 5 values from 0 to 1
print(f"Linspace (0 to 1, 5 values): {linspace_arr}")

# Identity matrix
identity = np.eye(3)  # 3x3 identity matrix
print(f"\nIdentity matrix (3x3):\n{identity}")

# ============================================================================
# 2. ARRAY PROPERTIES
# ============================================================================

print("\n" + "=" * 70)
print("2. ARRAY PROPERTIES")
print("=" * 70)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"Array:\n{arr}")
print(f"\nShape: {arr.shape}")  # (3, 4) - 3 rows, 4 columns
print(f"Size: {arr.size}")    # 12 - total elements
print(f"Ndim: {arr.ndim}")    # 2 - number of dimensions
print(f"Dtype: {arr.dtype}")  # int64 - data type

# ============================================================================
# 3. INDEXING (Accessing Elements)
# ============================================================================

print("\n" + "=" * 70)
print("3. INDEXING")
print("=" * 70)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Array:\n{arr}")

# 1D indexing
print(f"\nFirst row: {arr[0]}")
print(f"Second row: {arr[1]}")
print(f"Last row: {arr[-1]}")

# 2D indexing
print(f"\nElement at [0, 0]: {arr[0, 0]}")
print(f"Element at [1, 2]: {arr[1, 2]}")
print(f"Element at [2, 1]: {arr[2, 1]}")

# ============================================================================
# 4. SLICING (Getting Subsets)
# ============================================================================

print("\n" + "=" * 70)
print("4. SLICING")
print("=" * 70)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Original array:\n{arr}")

# Row slicing
print(f"\nFirst 2 rows:\n{arr[0:2]}")
print(f"Last row:\n{arr[-1:]}")

# Column slicing
print(f"\nFirst 2 columns:\n{arr[:, 0:2]}")
print(f"All rows, column 1:\n{arr[:, 1]}")

# Submatrix
print(f"\nSubmatrix [0:2, 1:3]:\n{arr[0:2, 1:3]}")

# ============================================================================
# 5. DATA TYPES (DTYPE)
# ============================================================================

print("\n" + "=" * 70)
print("5. DATA TYPES")
print("=" * 70)

# Integer array
int_arr = np.array([1, 2, 3, 4])
print(f"Integer array: {int_arr}, dtype: {int_arr.dtype}")

# Float array
float_arr = np.array([1.0, 2.5, 3.7])
print(f"Float array: {float_arr}, dtype: {float_arr.dtype}")

# Boolean array
bool_arr = np.array([True, False, True])
print(f"Boolean array: {bool_arr}, dtype: {bool_arr.dtype}")

# Specify dtype explicitly
explicit_int = np.array([1, 2, 3], dtype=np.float32)
print(f"Explicit float32: {explicit_int}, dtype: {explicit_int.dtype}")

# ============================================================================
# 6. RESHAPING ARRAYS
# ============================================================================

print("\n" + "=" * 70)
print("6. RESHAPING ARRAYS")
print("=" * 70)

arr = np.arange(12)  # [0, 1, 2, ..., 11]
print(f"Original (1D): {arr}, shape: {arr.shape}")

# Reshape to 2D
arr_2d = arr.reshape(3, 4)
print(f"\nReshaped to (3, 4):\n{arr_2d}")

# Reshape to 3D
arr_3d = arr.reshape(2, 3, 2)
print(f"\nReshaped to (2, 3, 2):\n{arr_3d}")

# Flatten back to 1D
arr_flat = arr_2d.flatten()
print(f"\nFlattened: {arr_flat}, shape: {arr_flat.shape}")

# ============================================================================
# 7. PRACTICAL EXAMPLE: Working with Data
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL EXAMPLE: Student Grades")
print("=" * 70)

# 3 students, 4 subjects
grades = np.array([
    [85, 90, 78, 92],  # Student 1
    [88, 92, 85, 89],  # Student 2
    [92, 88, 90, 95]   # Student 3
])

print(f"Grades (3 students, 4 subjects):\n{grades}")
print(f"\nStudent 1 grades: {grades[0]}")
print(f"Subject 1 grades: {grades[:, 0]}")
print(f"Student 2, Subject 3 grade: {grades[1, 2]}")

# Get first 2 students, first 3 subjects
subset = grades[0:2, 0:3]
print(f"\nFirst 2 students, first 3 subjects:\n{subset}")
