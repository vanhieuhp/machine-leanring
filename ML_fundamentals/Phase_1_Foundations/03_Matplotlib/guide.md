# Matplotlib Fundamentals Guide

## What is Matplotlib?

Matplotlib is the primary visualization library in Python. It provides:
- **Line plots** - show trends over time
- **Scatter plots** - show relationships between variables
- **Histograms** - show distributions
- **Bar charts** - compare categories
- **Box plots** - show statistical distributions
- **Heatmaps** - show 2D data patterns

## Why Matplotlib Matters for ML

Visualization is crucial for:
- Understanding data before modeling
- Debugging model performance
- Communicating results
- Identifying patterns and outliers
- Exploring relationships between features

## Learning Objectives

By the end of this section, you'll understand:
1. Creating basic plots
2. Customizing plots (labels, colors, legends)
3. Creating subplots
4. Different plot types
5. Saving figures

## Key Concepts

### 1. Figure and Axes

- **Figure** - the overall window/canvas
- **Axes** - the actual plot area where data is drawn

```python
fig, ax = plt.subplots()
ax.plot(x, y)
```

### 2. Plot Types

- **Line plot** - `plt.plot()` - trends
- **Scatter plot** - `plt.scatter()` - relationships
- **Histogram** - `plt.hist()` - distributions
- **Bar chart** - `plt.bar()` - categories
- **Box plot** - `plt.boxplot()` - statistics

### 3. Customization

- **Labels** - `xlabel()`, `ylabel()`, `title()`
- **Colors** - `color='red'`, `c='blue'`
- **Markers** - `marker='o'`, `marker='s'`
- **Line styles** - `linestyle='-'`, `linestyle='--'`
- **Legend** - `legend()`

## Study Path

1. **Start with**: `01_basic_plots.py`
   - Line plots
   - Scatter plots
   - Histograms
   - Basic customization

2. **Then**: `02_advanced_plots.py`
   - Subplots
   - Multiple plot types
   - Advanced customization
   - Saving figures

3. **Practice**: `exercises.py`
   - Create visualizations from data
   - Combine with Pandas

## Common Mistakes to Avoid

1. **Forgetting to show plots**
   - Use `plt.show()` to display
   - Or save with `plt.savefig()`

2. **Overcrowding plots**
   - Too many lines/points makes it unreadable
   - Use subplots for multiple plots

3. **Poor labeling**
   - Always add title, xlabel, ylabel
   - Use legends for multiple series

4. **Wrong plot type**
   - Line plot for trends
   - Scatter for relationships
   - Histogram for distributions
   - Bar for categories

5. **Ignoring figure size**
   - Use `figsize=(width, height)` for readability
   - Default is often too small

## Tips for Learning

- Always add labels and titles
- Use `plt.tight_layout()` to prevent overlap
- Experiment with different plot types
- Use colors and markers to distinguish data
- Save figures for reports

## Next Steps

After mastering Matplotlib:
- Combine with Pandas for data visualization
- Use Seaborn for statistical plots
- Create dashboards with multiple plots
- Communicate findings visually
