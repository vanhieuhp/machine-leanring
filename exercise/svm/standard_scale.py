import numpy as np
from sklearn.preprocessing import StandardScaler

x = np.array([20, 30, 40, 50, 60], dtype=float)

# Tự viết
mu = np.mean(x)
sigma = np.std(x)
x_scaled_manual = (x - mu) / sigma
print("mu:", mu, "sigma:", sigma, "x_scaled_manual:", x_scaled_manual)

# Sklearn
scaler = StandardScaler()
x_scaled_sklearn = scaler.fit_transform(x.reshape(-1, 1)).flatten()

print("Manual:", x_scaled_manual)
print("Sklearn:", x_scaled_sklearn)
print("Khớp nhau:", np.allclose(x_scaled_manual, x_scaled_sklearn))