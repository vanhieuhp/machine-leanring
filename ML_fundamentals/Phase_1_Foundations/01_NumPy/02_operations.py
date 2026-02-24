"""
NumPy Fundamentals - Part 2: Operations
========================================

This module covers:
- Element-wise operations
- Broadcasting
- Aggregation functions (sum, mean, std, etc.)
- Comparison operations
"""

import numpy as np

# ============================================================================
# 1. ELEMENT-WISE OPERATIONS
# ============================================================================

print("=" * 70)
print("1. ELEMENT-WISE OPERATIONS")
print("=" * 70)

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"Array a: {a}")
print(f"Array b: {b}")

# Arithmetic operations
print(f"\nAddition (a + b): {a + b}")
print(f"Subtraction (a - b): {a - b}")
print(f"Multiplication (a * b): {a * b}")
print(f"Division (b / a): {b / a}")
print(f"Power (a ** 2): {a ** 2}")
print(f"Modulo (b % a): {b % a}")

# ============================================================================
# 2. OPERATIONS WITH SCALARS
# ============================================================================

print("\n" + "=" * 70)
print("2. OPERATIONS WITH SCALARS")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")

print(f"\nAdd 10: {arr + 10}")
print(f"Multiply by 2: {arr * 2}")
print(f"Divide by 2: {arr / 2}")
print(f"Square root: {np.sqrt(arr)}")

# ============================================================================
# 3. BROADCASTING
# ============================================================================

print("\n" + "=" * 70)
print("3. BROADCASTING")
print("=" * 70)

# Broadcasting with different shapes
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Matrix (2x3):\n{matrix}")

scalar = 10
print(f"\nScalar: {scalar}")
print(f"Matrix + Scalar:\n{matrix + scalar}")

# Broadcasting row to matrix
row = np.array([1, 2, 3])
print(f"\nRow (1x3): {row}")
print(f"Matrix + Row:\n{matrix + row}")

# Broadcasting column to matrix
column = np.array([[10], [20]])
print(f"\nColumn (2x1):\n{column}")
print(f"Matrix + Column:\n{matrix + column}")

# ============================================================================
# 4. AGGREGATION FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("4. AGGREGATION FUNCTIONS")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")

print(f"\nSum: {np.sum(arr)}")
print(f"Mean: {np.mean(arr)}")
print(f"Median: {np.median(arr)}")
print(f"Std Dev: {np.std(arr)}")
print(f"Min: {np.min(arr)}")
print(f"Max: {np.max(arr)}")
print(f"Product: {np.prod(arr)}")

# 2D aggregation
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nMatrix:\n{matrix}")

print(f"\nSum all: {np.sum(matrix)}")
print(f"Sum along rows (axis=1): {np.sum(matrix, axis=1)}")
print(f"Sum along columns (axis=0): {np.sum(matrix, axis=0)}")

print(f"\nMean along rows (axis=1): {np.mean(matrix, axis=1)}")
print(f"Mean along columns (axis=0): {np.mean(matrix, axis=0)}")

# ============================================================================
# 5. COMPARISON OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("5. COMPARISON OPERATIONS")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")

print(f"\narr > 3: {arr > 3}")
print(f"arr == 3: {arr == 3}")
print(f"arr <= 2: {arr <= 2}")

# Using comparisons to filter
print(f"\nElements > 3: {arr[arr > 3]}")
print(f"Elements <= 2: {arr[arr <= 2]}")

# ============================================================================
# 6. LOGICAL OPERATIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. LOGICAL OPERATIONS")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")

# AND operation
condition1 = arr > 3
condition2 = arr < 8
result = np.logical_and(condition1, condition2)
print(f"\n(arr > 3) AND (arr < 8): {result}")
print(f"Elements: {arr[result]}")

# OR operation
condition1 = arr < 3
condition2 = arr > 8
result = np.logical_or(condition1, condition2)
print(f"\n(arr < 3) OR (arr > 8): {result}")
print(f"Elements: {arr[result]}")

# ============================================================================
# 7. PRACTICAL EXAMPLE: Data Normalization
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL EXAMPLE: Data Normalization")
print("=" * 70)

# Raw data (e.g., test scores)
scores = np.array([45, 67, 89, 92, 78, 56, 88, 95, 72, 81])
print(f"Raw scores: {scores}")

# Standardization (z-score normalization)
mean = np.mean(scores)
std = np.std(scores)
normalized = (scores - mean) / std

print(f"\nMean: {mean:.2f}")
print(f"Std Dev: {std:.2f}")
print(f"Normalized scores: {normalized}")

# Min-Max normalization (0 to 1)
min_val = np.min(scores)
max_val = np.max(scores)
min_max_normalized = (scores - min_val) / (max_val - min_val)

print(f"\nMin: {min_val}, Max: {max_val}")
print(f"Min-Max normalized: {min_max_normalized}")

# ============================================================================
# 8. PRACTICAL EXAMPLE: Feature Scaling
# ============================================================================

print("\n" + "=" * 70)
print("8. PRACTICAL EXAMPLE: Feature Scaling")
print("=" * 70)

# Dataset with different scales
# Feature 1: Age (0-100)
# Feature 2: Income (0-1000000)
features = np.array([
    [25, 50000],
    [35, 75000],
    [45, 100000],
    [55, 120000]
])

print(f"Original features:\n{features}")
print(f"Feature 1 (Age) range: {features[:, 0].min()} - {features[:, 0].max()}")
print(f"Feature 2 (Income) range: {features[:, 1].min()} - {features[:, 1].max()}")

# Standardize each feature
scaled_features = np.zeros_like(features, dtype=float)
for i in range(features.shape[1]):
    mean = np.mean(features[:, i])
    std = np.std(features[:, i])
    scaled_features[:, i] = (features[:, i] - mean) / std

print(f"\nScaled features:\n{scaled_features}")
print(f"Feature 1 mean: {np.mean(scaled_features[:, 0]):.6f}")
print(f"Feature 2 mean: {np.mean(scaled_features[:, 1]):.6f}")
