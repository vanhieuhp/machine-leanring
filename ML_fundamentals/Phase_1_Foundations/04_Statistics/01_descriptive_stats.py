"""
Statistics Fundamentals - Part 1: Descriptive Statistics
=========================================================

This module covers:
- Mean, median, mode
- Standard deviation and variance
- Percentiles and quartiles
- Skewness and kurtosis
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================================
# 1. CENTRAL TENDENCY
# ============================================================================

print("=" * 70)
print("1. CENTRAL TENDENCY (Mean, Median, Mode)")
print("=" * 70)

data = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 100])  # 100 is outlier
print(f"Data: {data}")

# Mean
mean = np.mean(data)
print(f"\nMean: {mean:.2f}")

# Median
median = np.median(data)
print(f"Median: {median:.2f}")

# Mode
mode_result = stats.mode(data, keepdims=True)
print(f"Mode: {mode_result.mode[0]}")

# Observation
print(f"\nObservation: Mean ({mean:.2f}) > Median ({median:.2f})")
print("This is because the outlier (100) pulls the mean up.")

# ============================================================================
# 2. SPREAD (Variance, Std Dev, Range)
# ============================================================================

print("\n" + "=" * 70)
print("2. SPREAD (Variance, Std Dev, Range)")
print("=" * 70)

data1 = np.array([10, 12, 14, 16, 18])
data2 = np.array([5, 10, 15, 20, 25])

print(f"Data 1: {data1}")
print(f"Data 2: {data2}")

# Both have same mean
print(f"\nMean 1: {np.mean(data1):.2f}")
print(f"Mean 2: {np.mean(data2):.2f}")

# But different spread
print(f"\nVariance 1: {np.var(data1):.2f}")
print(f"Variance 2: {np.var(data2):.2f}")

print(f"\nStd Dev 1: {np.std(data1):.2f}")
print(f"Std Dev 2: {np.std(data2):.2f}")

print(f"\nRange 1: {np.max(data1) - np.min(data1)}")
print(f"Range 2: {np.max(data2) - np.min(data2)}")

# ============================================================================
# 3. PERCENTILES AND QUARTILES
# ============================================================================

print("\n" + "=" * 70)
print("3. PERCENTILES AND QUARTILES")
print("=" * 70)

data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
print(f"Data: {data}")

# Percentiles
p25 = np.percentile(data, 25)
p50 = np.percentile(data, 50)
p75 = np.percentile(data, 75)

print(f"\n25th percentile (Q1): {p25}")
print(f"50th percentile (Q2/Median): {p50}")
print(f"75th percentile (Q3): {p75}")

# IQR (Interquartile Range)
iqr = p75 - p25
print(f"\nIQR (Q3 - Q1): {iqr}")

# ============================================================================
# 4. SKEWNESS AND KURTOSIS
# ============================================================================

print("\n" + "=" * 70)
print("4. SKEWNESS AND KURTOSIS")
print("=" * 70)

# Symmetric distribution
symmetric = np.random.normal(100, 15, 1000)
skew_sym = stats.skew(symmetric)
kurt_sym = stats.kurtosis(symmetric)

print(f"Symmetric distribution:")
print(f"  Skewness: {skew_sym:.3f} (close to 0)")
print(f"  Kurtosis: {kurt_sym:.3f}")

# Right-skewed distribution
right_skewed = np.concatenate([np.random.normal(50, 10, 900), np.random.normal(150, 20, 100)])
skew_right = stats.skew(right_skewed)
kurt_right = stats.kurtosis(right_skewed)

print(f"\nRight-skewed distribution:")
print(f"  Skewness: {skew_right:.3f} (positive)")
print(f"  Kurtosis: {kurt_right:.3f}")

# Left-skewed distribution
left_skewed = np.concatenate([np.random.normal(150, 10, 900), np.random.normal(50, 20, 100)])
skew_left = stats.skew(left_skewed)
kurt_left = stats.kurtosis(left_skewed)

print(f"\nLeft-skewed distribution:")
print(f"  Skewness: {skew_left:.3f} (negative)")
print(f"  Kurtosis: {kurt_left:.3f}")

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("5. SUMMARY STATISTICS")
print("=" * 70)

data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
print(f"Data: {data}")

print(f"\nCount: {len(data)}")
print(f"Mean: {np.mean(data):.2f}")
print(f"Std Dev: {np.std(data):.2f}")
print(f"Min: {np.min(data)}")
print(f"25%: {np.percentile(data, 25):.2f}")
print(f"50%: {np.percentile(data, 50):.2f}")
print(f"75%: {np.percentile(data, 75):.2f}")
print(f"Max: {np.max(data)}")

# ============================================================================
# 6. PRACTICAL EXAMPLE: Test Scores
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICAL EXAMPLE: Test Scores")
print("=" * 70)

# Two classes
class_a = np.array([65, 70, 72, 75, 78, 80, 82, 85, 88, 90])
class_b = np.array([55, 60, 70, 75, 80, 85, 90, 95, 98, 100])

print("Class A scores:", class_a)
print("Class B scores:", class_b)

print(f"\nClass A:")
print(f"  Mean: {np.mean(class_a):.2f}")
print(f"  Median: {np.median(class_a):.2f}")
print(f"  Std Dev: {np.std(class_a):.2f}")

print(f"\nClass B:")
print(f"  Mean: {np.mean(class_b):.2f}")
print(f"  Median: {np.median(class_b):.2f}")
print(f"  Std Dev: {np.std(class_b):.2f}")

print(f"\nInterpretation:")
print(f"  Class A: More consistent (lower std dev), lower average")
print(f"  Class B: More variable (higher std dev), higher average")

# ============================================================================
# 7. PRACTICAL EXAMPLE: Outlier Detection
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL EXAMPLE: Outlier Detection")
print("=" * 70)

data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 100])  # 100 is outlier
print(f"Data: {data}")

mean = np.mean(data)
std = np.std(data)

print(f"\nMean: {mean:.2f}")
print(f"Std Dev: {std:.2f}")

# Z-score method
z_scores = np.abs((data - mean) / std)
print(f"\nZ-scores: {z_scores}")

outliers = data[z_scores > 3]
print(f"Outliers (|z| > 3): {outliers}")

# IQR method
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nIQR method:")
print(f"  Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"  Lower bound: {lower_bound}, Upper bound: {upper_bound}")

outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
print(f"  Outliers: {outliers_iqr}")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("8. VISUALIZATION")
print("=" * 70)

# Create sample data
data = np.random.normal(100, 15, 1000)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.1f}')
axes[0, 0].axvline(np.median(data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(data):.1f}')
axes[0, 0].set_title('Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(data)
axes[0, 1].set_title('Box Plot')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Q-Q plot
stats.probplot(data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_data = np.sort(data)
cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axes[1, 1].plot(sorted_data, cumulative, linewidth=2)
axes[1, 1].set_title('Cumulative Distribution')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Cumulative Probability')

plt.tight_layout()
plt.show()
