"""
Clustering - Exercises
=====================

Practice problems for Clustering.
Solutions are provided at the bottom.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# ============================================================================
# EXERCISE 1: K-Means Basics
# ============================================================================

print("=" * 70)
print("EXERCISE 1: K-Means Basics")
print("=" * 70)

# 1.1 Create simple 2D dataset with 3 clusters
np.random.seed(42)
X1 = np.random.normal(loc=[0, 0], scale=1, size=(50, 2))
X2 = np.random.normal(loc=[5, 5], scale=1, size=(50, 2))
X3 = np.random.normal(loc=[0, 5], scale=1, size=(50, 2))
X = np.vstack([X1, X2, X3])

print(f"Data shape: {X.shape}")

# 1.2 Fit KMeans with k=3
# TODO: Fit KMeans

# 1.3 Get cluster labels
# TODO: Get labels

# 1.4 Get cluster centers
# TODO: Get centers

# 1.5 Calculate inertia (within-cluster sum of squares)
# TODO: Get inertia

# 1.6 Predict cluster for new point (2, 2)
new_point = np.array([[2, 2]])
# TODO: Predict

# ============================================================================
# EXERCISE 2: Elbow Method
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Elbow Method")
print("=" * 70)

# 2.1 Calculate inertia for k=1 to 10
k_range = range(1, 11)
inertias = []

for k in k_range:
    # TODO: Fit KMeans and get inertia

    # Append to list

    pass

# 2.2 Plot elbow curve
# TODO: Plot

# 2.3 Which k looks best from the elbow?

# ============================================================================
# EXERCISE 3: Silhouette Score
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Silhouette Score")
print("=" * 70)

# 3.1 Calculate silhouette scores for k=2 to 10
silhouette_scores = []

for k in range(2, 11):
    # TODO: Fit KMeans and calculate silhouette score

    # Append to list

    pass

# 3.2 Which k has highest silhouette score?
# TODO: Find best k

# 3.3 Plot silhouette scores
# TODO: Plot

# ============================================================================
# EXERCISE 4: Different Cluster Shapes
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Different Cluster Shapes")
print("=" * 70)

# 4.1 Create blob data (K-Means works well)
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)

# 4.2 Create moon data (K-Means struggles)
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

# 4.3 Create circle data
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# 4.4 Apply K-Means to each
for name, X_data, y_true in [("Blobs", X_blobs, y_blobs),
                               ("Moons", X_moons, y_moons),
                               ("Circles", X_circles, y_circles)]:
    # TODO: Fit KMeans with k=2

    # TODO: Calculate silhouette score

    print(f"{name}: Silhouette = {silhouette:.4f}")

# 4.5 Which algorithm works best for which shape?

# ============================================================================
# EXERCISE 5: Hierarchical Clustering
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Hierarchical Clustering")
print("=" * 70)

# 5.1 Create data
X_hier = np.random.randn(150, 2)

# 5.2 Fit AgglomerativeClustering with different linkages
linkages = ['ward', 'complete', 'average']

for linkage in linkages:
    # TODO: Fit hierarchical clustering

    # TODO: Get labels

    print(f"{linkage} linkage: {np.unique(labels)}")

# 5.3 Which linkage is best for the data?

# 5.4 Try with different n_clusters
for n in [2, 3, 4, 5]:
    # TODO: Fit and calculate silhouette

    # Print result

    pass

# ============================================================================
# EXERCISE 6: DBSCAN (Density-based)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: DBSCAN")
print("=" * 70)

# 6.1 Apply DBSCAN to moons data
# TODO: Fit DBSCAN

# 6.2 How many clusters found?
# TODO: Count clusters

# 6.3 How many noise points?
# TODO: Count noise points (-1 label)

# 6.4 Try different eps and min_samples
eps_values = [0.1, 0.3, 0.5]
min_samples_values = [3, 5, 10]

print("\nDBSCAN results:")
for eps in eps_values:
    for min_samp in min_samples_values:
        # TODO: Fit DBSCAN

        # TODO: Count clusters and noise

        # Print result

        pass

# ============================================================================
# EXERCISE 7: Real-world - Customer Segmentation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Customer Segmentation")
print("=" * 70)

# Simulated customer data
np.random.seed(42)
n_customers = 300

customers = pd.DataFrame({
    'annual_spending': np.concatenate([
        np.random.normal(1000, 200, 100),    # Low spenders
        np.random.normal(5000, 500, 100),     # Medium spenders
        np.random.normal(15000, 2000, 100)    # High spenders
    ]),
    'purchase_frequency': np.concatenate([
        np.random.normal(3, 1, 100),          # Low frequency
        np.random.normal(12, 2, 100),         # Medium frequency
        np.random.normal(25, 5, 100)          # High frequency
    ]),
    'age': np.random.randint(18, 70, n_customers)
})

# Shuffle
customers = customers.sample(frac=1).reset_index(drop=True)

print("Customer Dataset:")
print(customers.head())

# 7.1 Select features
# TODO: Select features

# 7.2 Scale features
# Hint: StandardScaler

# TODO: Scale

# 7.3 Find optimal k using elbow and silhouette
# TODO: Try k=2 to 8

# 7.4 Fit final model
# TODO: Fit with best k

# 7.5 Analyze clusters
# TODO: Get cluster statistics

# 7.6 Visualize
# TODO: Plot clusters

# ============================================================================
# EXERCISE 8: Cluster Evaluation Metrics
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Cluster Evaluation Metrics")
print("=" * 70)

# 8.1 Create labeled data
X_eval, y_eval = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# 8.2 Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_eval)

# 8.3 Calculate multiple metrics
# Silhouette Score: higher is better (-1 to 1)
silhouette = silhouette_score(X_eval, labels)

# Calinski-Harabasz Index: higher is better
ch_score = calinski_harabasz_score(X_eval, labels)

# Davies-Bouldin Index: lower is better
db_score = davies_bouldin_score(X_eval, labels)

print(f"Silhouette Score: {silhouette:.4f} (higher is better)")
print(f"Calinski-Harabasz: {ch_score:.4f} (higher is better)")
print(f"Davies-Bouldin: {db_score:.4f} (lower is better)")

# 8.4 Compare with different k
print("\nComparing different k:")
for k in [2, 3, 4, 5]:
    # TODO: Fit and calculate all metrics

    # Print results

    pass

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("\n--- EXERCISE 1: K-Means Basics ---")
np.random.seed(42)
X1 = np.random.normal(loc=[0, 0], scale=1, size=(50, 2))
X2 = np.random.normal(loc=[5, 5], scale=1, size=(50, 2))
X3 = np.random.normal(loc=[0, 5], scale=1, size=(50, 2))
X = np.vstack([X1, X2, X3])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_:.4f}")
print(f"Prediction for (2,2): {kmeans.predict([[2, 2]])[0]}")

print("\n--- EXERCISE 2: Elbow Method ---")
k_range = range(1, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- EXERCISE 3: Silhouette Score ---")
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

best_k = np.argmax(silhouette_scores) + 2
print(f"Best k: {best_k} (silhouette: {max(silhouette_scores):.4f})")

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- EXERCISE 4: Different Cluster Shapes ---")
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

for name, X_data, y_true in [("Blobs", X_blobs, y_blobs),
                               ("Moons", X_moons, y_moons),
                               ("Circles", X_circles, y_circles)]:
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_data)
    silhouette = silhouette_score(X_data, labels)
    print(f"{name}: Silhouette = {silhouette:.4f}")

print("\n--- EXERCISE 5: Hierarchical Clustering ---")
X_hier = np.random.randn(150, 2)

for linkage in ['ward', 'complete', 'average']:
    hc = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    labels = hc.fit_predict(X_hier)
    score = silhouette_score(X_hier, labels)
    print(f"{linkage} linkage: Silhouette = {score:.4f}")

print("\n--- EXERCISE 6: DBSCAN ---")
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_moons)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")

print("\nDBSCAN parameter search:")
for eps in [0.1, 0.3, 0.5]:
    for min_samp in [3, 5, 10]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samp)
        labels = dbscan.fit_predict(X_moons)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"eps={eps}, min_samples={min_samp}: clusters={n_clusters}, noise={n_noise}")

print("\n--- EXERCISE 7: Customer Segmentation ---")
np.random.seed(42)
n_customers = 300

customers = pd.DataFrame({
    'annual_spending': np.concatenate([
        np.random.normal(1000, 200, 100),
        np.random.normal(5000, 500, 100),
        np.random.normal(15000, 2000, 100)
    ]),
    'purchase_frequency': np.concatenate([
        np.random.normal(3, 1, 100),
        np.random.normal(12, 2, 100),
        np.random.normal(25, 5, 100)
    ]),
    'age': np.random.randint(18, 70, n_customers)
})
customers = customers.sample(frac=1).reset_index(drop=True)

X = customers[['annual_spending', 'purchase_frequency']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k
silhouette_scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = np.argmax(silhouette_scores) + 2
print(f"Best k: {best_k}")

# Final model
kmeans = KMeans(n_clusters=best_k, random_state=42)
customers['cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Statistics:")
print(customers.groupby('cluster')[['annual_spending', 'purchase_frequency', 'age']].mean())

# Visualize
plt.figure(figsize=(12, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=customers['cluster'], cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Spending (scaled)')
plt.ylabel('Purchase Frequency (scaled)')
plt.title('Customer Segments')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n--- EXERCISE 8: Cluster Evaluation Metrics ---")
X_eval, y_eval = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

print("Comparing different k:")
for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_eval)

    sil = silhouette_score(X_eval, labels)
    ch = calinski_harabasz_score(X_eval, labels)
    db = davies_bouldin_score(X_eval, labels)

    print(f"k={k}: Silhouette={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")
