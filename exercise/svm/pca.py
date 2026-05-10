import numpy as np

np.random.seed(42)
n = 100

toan = np.random.randn(n) * 10 + 50      # điểm Toán, trung bình 50
ly = toan + np.random.randn(n) * 3       # điểm Lý ≈ điểm Toán + nhiễu nhỏ
x = np.zeros((n, 2))

for i in range(n):
    x[i][0] = toan[i]
    x[i][1] = ly[i]

print("Shape: ", x.shape)
print("head: ", x[:5])

#calculate mean of each feature
mean = np.mean(x, axis=0)
print("Mean:", mean)

x_centered = np.zeros((n, 2))
for i in range(n):
    x_centered[i] = x[i] - mean

print("Centered:", x_centered[:5])

cov_matrix = np.zeros((2, 2))
for i in range(n):
    print(x_centered[i].shape)
    col = x_centered[i].reshape(2, 1)
    cov_matrix = cov_matrix + col.dot(col.T)

cov_matrix = cov_matrix / n

print("Covariance matrix:", cov_matrix)

a = np.array([3, 5])
b = a.reshape(-1, 1)


eigenvalues , eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("Eigenvalues sorted:", eigenvalues_sorted)
print("PC1:", eigenvectors_sorted[:, 0])
print("PC2:", eigenvectors_sorted[:, 1])

total_variance = np.sum(eigenvalues_sorted)
explained_variance_pc1 = eigenvalues_sorted[0] / total_variance * 100
explained_variance_pc2 = eigenvalues_sorted[1] / total_variance * 100

print("Total variance:", total_variance)
print("PC1 explains:", explained_variance_pc1, "%")
print("PC2 explains:", explained_variance_pc2, "%")

pc1 = eigenvectors_sorted[:, 0].reshape(2, 1, ) # shape (2, 1)

# project each data point onto pc1
x_pca = np.zeros((n, 1))
for i in range(n):
    x_pca[i] = np.dot(x_centered[i], pc1)

print("Original shape:", x.shape)
print("After PCA shape:", x_pca.shape)
print("First 5 projected values:\n", x_pca[:5])

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# sklearn uses standardization (divide by std), not just mean centering
# so results may differ slightly — but explained variance ratio should match
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

pca_sklearn = PCA(n_components=1)
X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)

print("Explained variance ratio:", pca_sklearn.explained_variance_ratio_)
print("Our result - PC1 explains: 97.54%")