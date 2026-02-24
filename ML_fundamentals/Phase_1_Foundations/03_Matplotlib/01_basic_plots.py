"""
Matplotlib Fundamentals - Part 1: Basic Plots
==============================================

This module covers:
- Line plots
- Scatter plots
- Histograms
- Bar charts
- Basic customization
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 1. LINE PLOTS
# ============================================================================

print("=" * 70)
print("1. LINE PLOTS")
print("=" * 70)

# Simple line plot
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Line Plot')
plt.grid(True)
plt.show()

# Multiple lines
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine and Cosine Functions')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================================
# 2. SCATTER PLOTS
# ============================================================================

print("\n" + "=" * 70)
print("2. SCATTER PLOTS")
print("=" * 70)

# Simple scatter plot
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.grid(True)
plt.show()

# Scatter with colors
categories = np.random.randint(0, 3, 100)
colors = ['red', 'blue', 'green']

plt.figure(figsize=(10, 6))
for i, color in enumerate(colors):
    mask = categories == i
    plt.scatter(x[mask], y[mask], label=f'Category {i}', color=color, alpha=0.6, s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Categories')
plt.legend()
plt.grid(True)
plt.show()

# ============================================================================
# 3. HISTOGRAMS
# ============================================================================

print("\n" + "=" * 70)
print("3. HISTOGRAMS")
print("=" * 70)

# Simple histogram
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normal Distribution')
plt.grid(True, alpha=0.3)
plt.show()

# Multiple histograms
data1 = np.random.normal(100, 15, 1000)
data2 = np.random.normal(110, 20, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data1, bins=30, alpha=0.5, label='Distribution 1', edgecolor='black')
plt.hist(data2, bins=30, alpha=0.5, label='Distribution 2', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Overlapping Histograms')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 4. BAR CHARTS
# ============================================================================

print("\n" + "=" * 70)
print("4. BAR CHARTS")
print("=" * 70)

# Simple bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='steelblue', edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# Grouped bar chart
categories = ['Q1', 'Q2', 'Q3', 'Q4']
sales_2023 = [100, 120, 140, 160]
sales_2024 = [110, 130, 150, 170]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, sales_2023, width, label='2023', color='steelblue')
plt.bar(x + width/2, sales_2024, width, label='2024', color='coral')
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.title('Sales Comparison')
plt.xticks(x, categories)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# ============================================================================
# 5. BOX PLOTS
# ============================================================================

print("\n" + "=" * 70)
print("5. BOX PLOTS")
print("=" * 70)

# Simple box plot
data = [np.random.normal(100, 15, 100) for _ in range(4)]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['A', 'B', 'C', 'D'])
plt.ylabel('Value')
plt.title('Box Plot')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# ============================================================================
# 6. CUSTOMIZATION
# ============================================================================

print("\n" + "=" * 70)
print("6. CUSTOMIZATION")
print("=" * 70)

# Customized plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(12, 7))
plt.plot(x, y, color='darkblue', linewidth=3, linestyle='--', marker='o',
         markersize=4, markerfacecolor='red', markeredgecolor='darkblue')
plt.xlabel('X axis', fontsize=12, fontweight='bold')
plt.ylabel('Y axis', fontsize=12, fontweight='bold')
plt.title('Customized Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.show()

# ============================================================================
# 7. PRACTICAL EXAMPLE: Stock Price
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL EXAMPLE: Stock Price")
print("=" * 70)

# Simulate stock price data
days = np.arange(1, 31)
price = 100 + np.cumsum(np.random.randn(30) * 2)

plt.figure(figsize=(12, 6))
plt.plot(days, price, marker='o', linewidth=2, markersize=6, color='darkgreen')
plt.fill_between(days, price, alpha=0.3, color='lightgreen')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.title('Stock Price Over 30 Days')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 8. PRACTICAL EXAMPLE: Distribution Comparison
# ============================================================================

print("\n" + "=" * 70)
print("8. PRACTICAL EXAMPLE: Distribution Comparison")
print("=" * 70)

# Test scores for two groups
group_a = np.random.normal(75, 10, 1000)
group_b = np.random.normal(80, 12, 1000)

plt.figure(figsize=(12, 6))
plt.hist(group_a, bins=30, alpha=0.6, label='Group A', color='blue', edgecolor='black')
plt.hist(group_b, bins=30, alpha=0.6, label='Group B', color='red', edgecolor='black')
plt.xlabel('Test Score')
plt.ylabel('Frequency')
plt.title('Test Score Distribution by Group')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# ============================================================================
# 9. PRACTICAL EXAMPLE: Correlation Visualization
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL EXAMPLE: Correlation Visualization")
print("=" * 70)

# Generate correlated data
np.random.seed(42)
x = np.random.randn(200)
y = 2 * x + np.random.randn(200) * 0.5

plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.5, s=50, color='steelblue')

# Add trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Correlation between X and Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
