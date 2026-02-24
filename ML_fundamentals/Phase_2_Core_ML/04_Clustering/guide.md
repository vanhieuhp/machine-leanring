# Clustering Guide

## What is Clustering?

Clustering groups similar data points together without predefined labels. It's an unsupervised learning technique.

## When to Use

- Customer segmentation
- Image compression
- Document organization
- Anomaly detection

## Key Concepts

### 1. K-Means Algorithm

Steps:
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

### 2. Distance Metrics

**Euclidean Distance**:
```
d = √(Σ(x_i - y_i)²)
```

**Manhattan Distance**:
```
d = Σ|x_i - y_i|
```

### 3. Choosing k

**Elbow Method**:
- Plot inertia vs k
- Look for "elbow" point
- k at elbow is optimal

**Silhouette Score**:
- Ranges from -1 to 1
- Higher is better
- Measures cluster cohesion

### 4. Other Clustering Methods

- **Hierarchical**: Build tree of clusters
- **DBSCAN**: Density-based clustering
- **Gaussian Mixture**: Probabilistic clustering

## Advantages

- Simple and fast
- Scalable to large datasets
- Works with any data type
- Easy to implement

## Disadvantages

- Must specify k in advance
- Sensitive to initialization
- Assumes spherical clusters
- Sensitive to outliers

## Study Files

1. `01_kmeans_basics.py` - K-Means algorithm
2. `02_clustering_evaluation.py` - Evaluate clusters
3. `03_hierarchical_clustering.py` - Alternative method
4. `exercises.py` - Practice problems
