from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# ---- hàm của bạn ----
def covariance_matrix(X):
    n = X.shape[0]
    X_centered = X - np.mean(X, axis=0)
    cov = (1/n) * X_centered.T @ X_centered
    return cov

def pca_from_scratch(X, n_components):
    cov = covariance_matrix(X)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :n_components]
    X_centered = X - np.mean(X, axis=0)
    X_projected = X_centered @ top_eigenvectors
    return X_projected, eigenvalues

# ---- load data ----
data = load_breast_cancer()
X = data.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- từ scratch ----
X_proj, eigenvalues = pca_from_scratch(X_scaled, n_components=2)
total = np.sum(eigenvalues)
ratio = eigenvalues[:2] / total

print("=== Từ scratch ===")
print(f"PC1: {ratio[0].real*100:.2f}%")
print(f"PC2: {ratio[1].real*100:.2f}%")

# ---- sklearn ----
pca_sk = PCA(n_components=2)
pca_sk.fit(X_scaled)
print("\n=== sklearn ===")
print(f"PC1: {pca_sk.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2: {pca_sk.explained_variance_ratio_[1]*100:.2f}%")