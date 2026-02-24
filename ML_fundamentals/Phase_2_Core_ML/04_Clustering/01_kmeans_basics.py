"""
Clustering - Part 1: K-Means Basics
====================================

This module covers:
- K-Means algorithm
- Elbow method
- Silhouette score
- Practical applications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. K-MEANS ALGORITHM
# ============================================================================

print("=" * 70)
print("1. K-MEANS ALGORITHM")
print("=" * 70)

# Generate sample data
np.random.seed(42)
n_samples = 300

# Cluster 1
X1 = np.random.randn(n_samples // 3, 2) + np.array([0, 0])

# Cluster 2
X2 = np.random.randn(n_samples // 3, 2) + np.array([5, 5])

# Cluster 3
X3 = np.random.randn(n_samples // 3, 2) + np.array([0, 5])

X = np.vstack([X1, X2, X3])

print(f"Data shape: {X.shape}")
print(f"Number of clusters: 3")

# Train K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

print(f"Inertia: {kmeans.inertia_:.4f}")
print(f"Centroids:\n{centroids}")

# ============================================================================
# 2. ELBOW METHOD
# ============================================================================

print("\n" + "=" * 70)
print("2. ELBOW METHOD")
print("=" * 70)

inertias = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans_k.fit_predict(X)
    inertias.append(kmeans_k.inertia_)

    if k > 1:
        sil_score = silhouette_score(X, labels_k)
        silhouette_scores.append(sil_score)
    else:
        silhouette_scores.append(0)

    print(f"k={k}: Inertia={kmeans_k.inertia_:.4f}, Silhouette={silhouette_scores[-1]:.4f}")

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("3. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Data with clusters
ax = axes[0, 0]
scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('K-Means Clustering (k=3)')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Cluster')

# Plot 2: Elbow curve
ax = axes[0, 1]
ax.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
ax.grid(True, alpha=0.3)
ax.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Optimal k=3')
ax.legend()

# Plot 3: Silhouette score
ax = axes[1, 0]
ax.plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='green')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score vs k')
ax.grid(True, alpha=0.3)
ax.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Optimal k=3')
ax.legend()

# Plot 4: Silhouette plot for k=3
ax = axes[1, 1]
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_3 = kmeans_3.fit_predict(X)
silhouette_vals = silhouette_samples(X, labels_3)

y_lower = 10
for i in range(3):
    cluster_silhouette_vals = silhouette_vals[labels_3 == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                     alpha=0.7, label=f'Cluster {i}')
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title('Silhouette Plot (k=3)')
ax.axvline(x=silhouette_score(X, labels_3), color='red', linestyle='--', label='Average')
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# 4. SILHOUETTE ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("4. SILHOUETTE ANALYSIS")
print("=" * 70)

print(f"Silhouette score for k=3: {silhouette_score(X, labels_3):.4f}")
print(f"Silhouette scores per sample (first 10):")
print(silhouette_vals[:10])

# ============================================================================
# 5. PRACTICAL EXAMPLE: CUSTOMER SEGMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("5. PRACTICAL EXAMPLE: CUSTOMER SEGMENTATION")
print("=" * 70)

# Create customer data
np.random.seed(42)
n_customers = 300

# Segment 1: Budget customers
spending_1 = np.random.normal(500, 100, n_customers // 3)
frequency_1 = np.random.normal(5, 2, n_customers // 3)

# Segment 2: Regular customers
spending_2 = np.random.normal(2000, 300, n_customers // 3)
frequency_2 = np.random.normal(15, 3, n_customers // 3)

# Segment 3: Premium customers
spending_3 = np.random.normal(5000, 500, n_customers // 3)
frequency_3 = np.random.normal(30, 5, n_customers // 3)

X_customers = np.vstack([
    np.column_stack([spending_1, frequency_1]),
    np.column_stack([spending_2, frequency_2]),
    np.column_stack([spending_3, frequency_3])
])

# Scale features
scaler = StandardScaler()
X_customers_scaled = scaler.fit_transform(X_customers)

# Find optimal k
inertias_cust = []
for k in range(1, 8):
    kmeans_cust = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_cust.fit(X_customers_scaled)
    inertias_cust.append(kmeans_cust.inertia_)

# Train with optimal k
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_final = kmeans_final.fit_predict(X_customers_scaled)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_customers[:, 0], X_customers[:, 1], c=labels_final, cmap='viridis', alpha=0.6, s=50)
centroids_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2)
plt.xlabel('Annual Spending ($)')
plt.ylabel('Purchase Frequency')
plt.title('Customer Segmentation')
plt.colorbar(scatter, label='Segment')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, 8), inertias_cust, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Customer Data')
plt.axvline(x=3, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Segment statistics
print("\nSegment Statistics:")
for i in range(3):
    segment_data = X_customers[labels_final == i]
    print(f"\nSegment {i}:")
    print(f"  Size: {len(segment_data)} customers")
    print(f"  Avg Spending: ${segment_data[:, 0].mean():.2f}")
    print(f"  Avg Frequency: {segment_data[:, 1].mean():.1f}")
