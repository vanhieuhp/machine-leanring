"""
NumPy Fundamentals - Part 3: Matrix Operations
===============================================

This module covers:
- Dot products
- Matrix multiplication
- Transpose
- Determinant and inverse
- Solving linear systems
"""

import numpy as np

# ============================================================================
# 1. DOT PRODUCT
# ============================================================================

print("=" * 70)
print("1. DOT PRODUCT")
print("=" * 70)

# Dot product of two 1D arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"Vector a: {a}")
print(f"Vector b: {b}")

dot_product = np.dot(a, b)
print(f"\nDot product (a · b): {dot_product}")
print(f"Calculation: 1*4 + 2*5 + 3*6 = {1*4 + 2*5 + 3*6}")

# ============================================================================
# 2. MATRIX MULTIPLICATION
# ============================================================================

print("\n" + "=" * 70)
print("2. MATRIX MULTIPLICATION")
print("=" * 70)

# Matrix A (2x3) and Matrix B (3x2)
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

print(f"Matrix A (2x3):\n{A}")
print(f"\nMatrix B (3x2):\n{B}")

# Matrix multiplication
C = np.dot(A, B)
print(f"\nA · B (2x2):\n{C}")

# Alternative: using @ operator
C_alt = A @ B
print(f"\nUsing @ operator:\n{C_alt}")

# ============================================================================
# 3. TRANSPOSE
# ============================================================================

print("\n" + "=" * 70)
print("3. TRANSPOSE")
print("=" * 70)

A = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original matrix A (2x3):\n{A}")

A_T = A.T
print(f"\nTranspose A.T (3x2):\n{A_T}")

# Transpose of transpose
print(f"\n(A.T).T:\n{A_T.T}")

# ============================================================================
# 4. ELEMENT-WISE MULTIPLICATION vs MATRIX MULTIPLICATION
# ============================================================================

print("\n" + "=" * 70)
print("4. ELEMENT-WISE vs MATRIX MULTIPLICATION")
print("=" * 70)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"\nMatrix B:\n{B}")

# Element-wise multiplication (*)
element_wise = A * B
print(f"\nElement-wise (A * B):\n{element_wise}")

# Matrix multiplication (@)
matrix_mult = A @ B
print(f"\nMatrix multiplication (A @ B):\n{matrix_mult}")

# ============================================================================
# 5. IDENTITY AND INVERSE
# ============================================================================

print("\n" + "=" * 70)
print("5. IDENTITY AND INVERSE")
print("=" * 70)

# Identity matrix
I = np.eye(3)
print(f"Identity matrix (3x3):\n{I}")

# Square matrix
A = np.array([[1, 2], [3, 4]])
print(f"\nMatrix A:\n{A}")

# Inverse
A_inv = np.linalg.inv(A)
print(f"\nInverse of A:\n{A_inv}")

# Verify: A @ A_inv = I
result = A @ A_inv
print(f"\nA @ A_inv (should be close to I):\n{result}")

# ============================================================================
# 6. DETERMINANT
# ============================================================================

print("\n" + "=" * 70)
print("6. DETERMINANT")
print("=" * 70)

A = np.array([[1, 2], [3, 4]])
print(f"Matrix A:\n{A}")

det = np.linalg.det(A)
print(f"\nDeterminant: {det}")
print(f"Calculation: 1*4 - 2*3 = {1*4 - 2*3}")

# ============================================================================
# 7. RANK AND TRACE
# ============================================================================

print("\n" + "=" * 70)
print("7. RANK AND TRACE")
print("=" * 70)

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix A:\n{A}")

rank = np.linalg.matrix_rank(A)
print(f"\nRank: {rank}")

trace = np.trace(A)
print(f"Trace (sum of diagonal): {trace}")
print(f"Calculation: 1 + 5 + 9 = {1 + 5 + 9}")

# ============================================================================
# 8. SOLVING LINEAR SYSTEMS
# ============================================================================

print("\n" + "=" * 70)
print("8. SOLVING LINEAR SYSTEMS")
print("=" * 70)

# Solve: Ax = b
# 2x + 3y = 8
# 4x + y = 10

A = np.array([[2, 3], [4, 1]])
b = np.array([8, 10])

print(f"System: Ax = b")
print(f"A:\n{A}")
print(f"b: {b}")

x = np.linalg.solve(A, b)
print(f"\nSolution x: {x}")
print(f"x = {x[0]:.2f}, y = {x[1]:.2f}")

# Verify
result = A @ x
print(f"\nVerification (A @ x): {result}")
print(f"Should equal b: {b}")

# ============================================================================
# 9. EIGENVALUES AND EIGENVECTORS
# ============================================================================

print("\n" + "=" * 70)
print("9. EIGENVALUES AND EIGENVECTORS")
print("=" * 70)

A = np.array([[4, -2], [1, 1]])
print(f"Matrix A:\n{A}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors:\n{eigenvectors}")

# ============================================================================
# 10. PRACTICAL EXAMPLE: Linear Regression (Manual)
# ============================================================================

print("\n" + "=" * 70)
print("10. PRACTICAL EXAMPLE: Linear Regression")
print("=" * 70)

# Simple linear regression: y = mx + b
# Data points
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

print(f"X (features):\n{X.flatten()}")
print(f"y (target): {y}")

# Add bias term (column of 1s)
X_with_bias = np.column_stack([np.ones(len(X)), X])
print(f"\nX with bias:\n{X_with_bias}")

# Normal equation: β = (X^T X)^-1 X^T y
XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y
beta = np.linalg.inv(XTX) @ XTy

print(f"\nCoefficients (bias, slope): {beta}")
print(f"Equation: y = {beta[0]:.2f} + {beta[1]:.2f}x")

# Predictions
y_pred = X_with_bias @ beta
print(f"\nPredictions: {y_pred}")
print(f"Actual: {y}")

# ============================================================================
# 11. PRACTICAL EXAMPLE: Covariance Matrix
# ============================================================================

print("\n" + "=" * 70)
print("11. PRACTICAL EXAMPLE: Covariance Matrix")
print("=" * 70)

# Dataset: Age and Income
data = np.array([
    [25, 50000],
    [35, 75000],
    [45, 100000],
    [55, 120000],
    [65, 140000]
])

print(f"Data (Age, Income):\n{data}")

# Calculate covariance matrix
cov_matrix = np.cov(data.T)
print(f"\nCovariance matrix:\n{cov_matrix}")

# Correlation matrix
corr_matrix = np.corrcoef(data.T)
print(f"\nCorrelation matrix:\n{corr_matrix}")
