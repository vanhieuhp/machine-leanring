"""
Matplotlib Fundamentals - Part 2: Advanced Plots
=================================================

This module covers:
- Subplots
- Multiple plot types
- Advanced customization
- Saving figures
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 1. SUBPLOTS
# ============================================================================

print("=" * 70)
print("1. SUBPLOTS")
print("=" * 70)

# 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), color='blue')
axes[0, 0].set_title('Sine Function')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter plot
np.random.seed(42)
axes[0, 1].scatter(np.random.randn(100), np.random.randn(100), alpha=0.5)
axes[0, 1].set_title('Random Scatter')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Histogram
axes[1, 0].hist(np.random.normal(100, 15, 1000), bins=30, edgecolor='black')
axes[1, 0].set_title('Normal Distribution')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 1].bar(categories, values, color='steelblue')
axes[1, 1].set_title('Bar Chart')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 2. DIFFERENT SUBPLOT SIZES
# ============================================================================

print("\n" + "=" * 70)
print("2. DIFFERENT SUBPLOT SIZES")
print("=" * 70)

fig = plt.figure(figsize=(12, 8))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Large plot (top left, 2x2)
ax1 = fig.add_subplot(gs[0:2, 0:2])
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), linewidth=2)
ax1.set_title('Large Plot')
ax1.grid(True, alpha=0.3)

# Small plots (right side)
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(np.random.normal(100, 15, 1000), bins=20, edgecolor='black')
ax2.set_title('Histogram')

ax3 = fig.add_subplot(gs[1, 2])
ax3.scatter(np.random.randn(50), np.random.randn(50), alpha=0.5)
ax3.set_title('Scatter')

# Bottom plots
ax4 = fig.add_subplot(gs[2, :])
categories = ['Q1', 'Q2', 'Q3', 'Q4']
values = [100, 120, 140, 160]
ax4.bar(categories, values, color='steelblue')
ax4.set_title('Quarterly Data')
ax4.grid(True, alpha=0.3, axis='y')

plt.show()

# ============================================================================
# 3. FIGURE CUSTOMIZATION
# ============================================================================

print("\n" + "=" * 70)
print("3. FIGURE CUSTOMIZATION")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 7))

x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, linewidth=3, color='darkblue', label='sin(x)')
ax.fill_between(x, y, alpha=0.3, color='lightblue')

# Customize axes
ax.set_xlabel('X axis', fontsize=12, fontweight='bold')
ax.set_ylabel('Y axis', fontsize=12, fontweight='bold')
ax.set_title('Customized Plot', fontsize=14, fontweight='bold', pad=20)

# Customize grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Customize ticks
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(-1, 1.5, 0.5))

# Add legend
ax.legend(fontsize=11, loc='upper right')

# Set limits
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

plt.show()

# ============================================================================
# 4. MULTIPLE AXES
# ============================================================================

print("\n" + "=" * 70)
print("4. MULTIPLE AXES")
print("=" * 70)

fig, ax1 = plt.subplots(figsize=(12, 6))

# First y-axis
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
color = 'tab:blue'
ax1.set_xlabel('X')
ax1.set_ylabel('sin(x)', color=color)
ax1.plot(x, y1, color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# Second y-axis
ax2 = ax1.twinx()
y2 = np.exp(x / 10)
color = 'tab:red'
ax2.set_ylabel('exp(x/10)', color=color)
ax2.plot(x, y2, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

# ============================================================================
# 5. SAVING FIGURES
# ============================================================================

print("\n" + "=" * 70)
print("5. SAVING FIGURES")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), linewidth=2, color='blue')
ax.set_title('Sine Function')
ax.grid(True, alpha=0.3)

# Save as PNG
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
print("Saved as plot.png")

# Save as PDF
plt.savefig('plot.pdf', bbox_inches='tight')
print("Saved as plot.pdf")

plt.show()

# ============================================================================
# 6. PRACTICAL EXAMPLE: Time Series
# ============================================================================

print("\n" + "=" * 70)
print("6. PRACTICAL EXAMPLE: Time Series")
print("=" * 70)

# Generate time series data
dates = np.arange(100)
values = 100 + np.cumsum(np.random.randn(100) * 2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Time series
ax1.plot(dates, values, linewidth=2, color='darkgreen')
ax1.fill_between(dates, values, alpha=0.3, color='lightgreen')
ax1.set_ylabel('Value')
ax1.set_title('Time Series Data')
ax1.grid(True, alpha=0.3)

# Plot 2: Daily changes
changes = np.diff(values)
colors = ['green' if x > 0 else 'red' for x in changes]
ax2.bar(dates[1:], changes, color=colors, alpha=0.7)
ax2.set_xlabel('Day')
ax2.set_ylabel('Change')
ax2.set_title('Daily Changes')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

# ============================================================================
# 7. PRACTICAL EXAMPLE: Comparison Dashboard
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL EXAMPLE: Comparison Dashboard")
print("=" * 70)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Sales trend
ax1 = fig.add_subplot(gs[0, :])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [100, 120, 140, 160, 180, 200]
ax1.plot(months, sales, marker='o', linewidth=2, markersize=8, color='darkblue')
ax1.fill_between(range(len(months)), sales, alpha=0.3, color='lightblue')
ax1.set_title('Sales Trend', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Product distribution
ax2 = fig.add_subplot(gs[1, 0])
products = ['A', 'B', 'C', 'D']
quantities = [30, 25, 20, 25]
ax2.pie(quantities, labels=products, autopct='%1.1f%%', startangle=90)
ax2.set_title('Product Distribution', fontsize=12, fontweight='bold')

# Regional sales
ax3 = fig.add_subplot(gs[1, 1])
regions = ['North', 'South', 'East', 'West']
sales_by_region = [150, 120, 180, 140]
ax3.bar(regions, sales_by_region, color=['red', 'blue', 'green', 'orange'])
ax3.set_title('Sales by Region', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Performance metrics
ax4 = fig.add_subplot(gs[2, :])
metrics = ['Revenue', 'Profit', 'Growth', 'Satisfaction']
values = [85, 72, 90, 88]
colors_metrics = ['green' if x > 80 else 'orange' for x in values]
ax4.barh(metrics, values, color=colors_metrics)
ax4.set_xlabel('Score')
ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')
ax4.set_xlim(0, 100)
ax4.grid(True, alpha=0.3, axis='x')

plt.show()

# Clean up
import os
if os.path.exists('plot.png'):
    os.remove('plot.png')
if os.path.exists('plot.pdf'):
    os.remove('plot.pdf')
