"""
NumPy Exercises - Practice Problems
====================================

Complete these exercises to solidify your NumPy knowledge.
Solutions are provided at the bottom.
"""

import numpy as np

# ============================================================================
# EXERCISE 1: Array Creation and Properties
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Array Creation and Properties")
print("=" * 70)

# 1.1 Create a 1D array with values 0 to 9
# TODO: Create array
arr1 = None

# 1.2 Create a 3x3 matrix of zeros
# TODO: Create matrix
matrix_zeros = None

# 1.3 Create a 2x4 matrix of ones
# TODO: Create matrix
matrix_ones = None

# 1.4 Create a 5x5 identity matrix
# TODO: Create matrix
identity = None

# 1.5 Create an array with 10 evenly spaced values from 0 to 1
# TODO: Create array
linspace_arr = None

# ============================================================================
# EXERCISE 2: Indexing and Slicing
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Indexing and Slicing")
print("=" * 70)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 2.1 Get the first row
# TODO: Get first row
first_row = None

# 2.2 Get the last column
# TODO: Get last column
last_col = None

# 2.3 Get elements where row > 0 and column > 1
# TODO: Get submatrix
submatrix = None

# 2.4 Get all elements greater than 6
# TODO: Filter elements
filtered = None

# ============================================================================
# EXERCISE 3: Operations
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Operations")
print("=" * 70)

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# 3.1 Add a and b
# TODO: Add arrays
sum_ab = None

# 3.2 Multiply a by 5
# TODO: Multiply
mult_a = None

# 3.3 Divide b by a
# TODO: Divide
div_ba = None

# 3.4 Calculate element-wise square of a
# TODO: Square
square_a = None

# ============================================================================
# EXERCISE 4: Aggregation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Aggregation")
print("=" * 70)

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# 4.1 Calculate sum
# TODO: Sum
total = None

# 4.2 Calculate mean
# TODO: Mean
average = None

# 4.3 Calculate standard deviation
# TODO: Std dev
std_dev = None

# 4.4 Find minimum and maximum
# TODO: Min and max
min_val = None
max_val = None

# ============================================================================
# EXERCISE 5: Matrix Operations
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Matrix Operations")
print("=" * 70)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 5.1 Transpose A
# TODO: Transpose
A_T = None

# 5.2 Matrix multiply A and B
# TODO: Matrix multiply
AB = None

# 5.3 Element-wise multiply A and B
# TODO: Element-wise multiply
A_elem_B = None

# ============================================================================
# EXERCISE 6: Normalization
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Normalization")
print("=" * 70)

scores = np.array([45, 67, 89, 92, 78, 56, 88, 95, 72, 81])

# 6.1 Standardize scores (z-score normalization)
# TODO: Standardize
mean = None
std = None
standardized = None

# 6.2 Min-Max normalize scores (0 to 1)
# TODO: Min-Max normalize
min_val = None
max_val = None
min_max_norm = None

# ============================================================================
# EXERCISE 7: Broadcasting
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Broadcasting")
print("=" * 70)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
column = np.array([[100], [200]])

# 7.1 Add row to each row of matrix
# TODO: Add row
result1 = None

# 7.2 Add column to each column of matrix
# TODO: Add column
result2 = None

# ============================================================================
# EXERCISE 8: Filtering
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Filtering")
print("=" * 70)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 8.1 Get all elements greater than 5
# TODO: Filter > 5
greater_than_5 = None

# 8.2 Get all even numbers
# TODO: Filter even
even_numbers = None

# 8.3 Get elements between 3 and 7 (inclusive)
# TODO: Filter range
between_3_7 = None

# ============================================================================
# EXERCISE 9: Reshaping
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9: Reshaping")
print("=" * 70)

arr = np.arange(12)

# 9.1 Reshape to 3x4
# TODO: Reshape
reshaped_3x4 = None

# 9.2 Reshape to 2x2x3
# TODO: Reshape
reshaped_2x2x3 = None

# 9.3 Flatten a 3x4 matrix back to 1D
# TODO: Flatten
matrix_3x4 = np.arange(12).reshape(3, 4)
flattened = None

# ============================================================================
# EXERCISE 10: Real-world Problem
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 10: Real-world Problem - Student Grades")
print("=" * 70)

# Student grades: 4 students, 3 subjects
grades = np.array([
    [85, 90, 78],
    [88, 92, 85],
    [92, 88, 90],
    [78, 85, 88]
])

# 10.1 Calculate average grade for each student
# TODO: Average per student
student_avg = None

# 10.2 Calculate average grade for each subject
# TODO: Average per subject
subject_avg = None

# 10.3 Find the highest grade
# TODO: Max grade
highest_grade = None

# 10.4 Find which student has the highest average
# TODO: Best student
best_student_idx = None

# 10.5 Normalize all grades to 0-1 range
# TODO: Normalize
grades_normalized = None

# ============================================================================
# SOLUTIONS (Uncomment to see)
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("\n1.1 - 1.5: Array Creation")
print(f"1.1: {np.arange(10)}")
print(f"1.2:\n{np.zeros((3, 3))}")
print(f"1.3:\n{np.ones((2, 4))}")
print(f"1.4:\n{np.eye(5)}")
print(f"1.5: {np.linspace(0, 1, 10)}")

print("\n2.1 - 2.4: Indexing and Slicing")
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"2.1 First row: {arr[0]}")
print(f"2.2 Last column: {arr[:, -1]}")
print(f"2.3 Submatrix [1:, 2:]: \n{arr[1:, 2:]}")
print(f"2.4 Elements > 6: {arr[arr > 6]}")

print("\n3.1 - 3.4: Operations")
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
print(f"3.1 a + b: {a + b}")
print(f"3.2 a * 5: {a * 5}")
print(f"3.3 b / a: {b / a}")
print(f"3.4 a^2: {a ** 2}")

print("\n4.1 - 4.4: Aggregation")
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"4.1 Sum: {np.sum(arr)}")
print(f"4.2 Mean: {np.mean(arr)}")
print(f"4.3 Std Dev: {np.std(arr)}")
print(f"4.4 Min: {np.min(arr)}, Max: {np.max(arr)}")

print("\n5.1 - 5.3: Matrix Operations")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"5.1 A.T:\n{A.T}")
print(f"5.2 A @ B:\n{A @ B}")
print(f"5.3 A * B:\n{A * B}")

print("\n6.1 - 6.2: Normalization")
scores = np.array([45, 67, 89, 92, 78, 56, 88, 95, 72, 81])
mean = np.mean(scores)
std = np.std(scores)
standardized = (scores - mean) / std
print(f"6.1 Standardized: {standardized}")
min_val = np.min(scores)
max_val = np.max(scores)
min_max_norm = (scores - min_val) / (max_val - min_val)
print(f"6.2 Min-Max normalized: {min_max_norm}")

print("\n7.1 - 7.2: Broadcasting")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
column = np.array([[100], [200]])
print(f"7.1 matrix + row:\n{matrix + row}")
print(f"7.2 matrix + column:\n{matrix + column}")

print("\n8.1 - 8.3: Filtering")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"8.1 > 5: {arr[arr > 5]}")
print(f"8.2 Even: {arr[arr % 2 == 0]}")
print(f"8.3 Between 3-7: {arr[(arr >= 3) & (arr <= 7)]}")

print("\n9.1 - 9.3: Reshaping")
arr = np.arange(12)
print(f"9.1 Reshape 3x4:\n{arr.reshape(3, 4)}")
print(f"9.2 Reshape 2x2x3:\n{arr.reshape(2, 2, 3)}")
matrix_3x4 = np.arange(12).reshape(3, 4)
print(f"9.3 Flattened: {matrix_3x4.flatten()}")

print("\n10.1 - 10.5: Student Grades")
grades = np.array([[85, 90, 78], [88, 92, 85], [92, 88, 90], [78, 85, 88]])
print(f"10.1 Student averages: {np.mean(grades, axis=1)}")
print(f"10.2 Subject averages: {np.mean(grades, axis=0)}")
print(f"10.3 Highest grade: {np.max(grades)}")
print(f"10.4 Best student index: {np.argmax(np.mean(grades, axis=1))}")
min_g = np.min(grades)
max_g = np.max(grades)
print(f"10.5 Normalized:\n{(grades - min_g) / (max_g - min_g)}")
