from sklearn.cluster import KMeans

sklearn_model = KMeans(n_clusters=3, n_init=10, random_state=42)
sklearn_model.fit(points)

print(f"Inertia sklearn: {sklearn_model.inertia_:.2f}")