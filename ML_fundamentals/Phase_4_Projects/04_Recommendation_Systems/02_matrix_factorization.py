"""
Recommendation Systems - Part 2: Matrix Factorization
===================================================

This module covers:
- SVD (Singular Value Decomposition)
- NMF (Non-negative Matrix Factorization)
- ALS (Alternating Least Squares)
- FunkSVD
- Implementation from scratch and with libraries

Based on: Movie Recommendation (MovieLens)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. MATRIX FACTORIZATION CONCEPT
# ============================================================================

print("=" * 70)
print("1. MATRIX FACTORIZATION CONCEPT")
print("=" * 70)

print("""
Matrix Factorization:
====================

Goal: Decompose user-item matrix R into two lower-dimensional matrices:

R ≈ U × V^T

Where:
- R: (m × n) user-item matrix
- U: (m × k) user latent factors
- V: (n × k) item latent factors
- k: number of latent factors (hidden dimensions)

This captures underlying patterns in the data.

Why it works:
- Users have latent preferences
- Items have latent features
- Dot product predicts ratings
""")

# Create sample data
np.random.seed(42)

n_users = 20
n_items = 30
n_factors = 5  # latent factors

# Create latent factors
U = np.random.rand(n_users, n_factors)  # User preferences
V = np.random.rand(n_items, n_factors)  # Item features

# Add some bias
user_bias = np.random.rand(n_users) * 0.5
item_bias = np.random.rand(n_items) * 0.5

# Create rating matrix
R_true = U @ V.T + user_bias + item_bias

# Add noise and binarize
R = R_true + np.random.randn(n_users, n_items) * 0.3
R = np.clip(R, 1, 5)

# Make sparse (remove some ratings)
mask = np.random.rand(n_users, n_items) > 0.7
R_sparse = R.copy()
R_sparse[~mask] = 0

print(f"User-Item Matrix: {R_sparse.shape}")
print(f"Users: {n_users}, Items: {n_items}")
print(f"Latent factors: {n_factors}")

# ============================================================================
# 2. SVD (SINGULAR VALUE DECOMPOSITION)
# ============================================================================

print("\n" + "=" * 70)
print("2. SVD (SINGULAR VALUE DECOMPOSITION)")
print("=" * 70)

# -------------------------------------------------------------------------
# 2.1 Basic SVD
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.1 Basic SVD")
print("-" * 50)

# Replace zeros with mean for SVD
R_filled = R_sparse.copy()
R_filled[R_filled == 0] = R_filled[R_filled > 0].mean()

# SVD decomposition
U_svd, s, Vt = np.linalg.svd(R_filled, full_matrices=False)

print(f"SVD Results:")
print(f"  U shape: {U_svd.shape}")
print(f"  Singular values: {len(s)}")
print(f"  Vt shape: {Vt.shape}")

# Reconstruct with k factors
k = 10
R_reconstructed = U_svd[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

print(f"\nReconstructed matrix (k={k}):")
print(f"  Original non-zero RMSE: {np.sqrt(mean_squared_error(R[R>0], R_reconstructed[R>0])):.3f}")

# -------------------------------------------------------------------------
# 2.2 Truncated SVD (for sparse matrices)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.2 Truncated SVD (Sparse)")
print("-" * 50)

# Create sparse matrix
sparse_matrix = csr_matrix(R_filled)

# Truncated SVD
n_components = 10
svd = TruncatedSVD(n_components=n_components, random_state=42)
U_truncated = svd.fit_transform(sparse_matrix)
V_truncated = svd.components_.T

print(f"Truncated SVD:")
print(f"  Explained variance: {svd.explained_variance_ratio_.sum()*100:.1f}%")
print(f"  U shape: {U_truncated.shape}")
print(f"  V shape: {V_truncated.shape}")

# Reconstruct
R_pred = U_truncated @ V_truncated.T
print(f"  Reconstruction error: {np.sqrt(mean_squared_error(R[R>0], R_pred[R>0])):.3f}")

# -------------------------------------------------------------------------
# 2.3 SVD with scikit-learn
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.3 Using sklearn TruncatedSVD")
print("-" * 50)

# Fit on user-item matrix
svd_model = TruncatedSVD(n_components=5, random_state=42)

# Transform
user_factors = svd_model.fit_transform(R_filled)
item_factors = svd_model.components_.T

print(f"User factors shape: {user_factors.shape}")
print(f"Item factors shape: {item_factors.shape}")

# Predict
predictions = user_factors @ item_factors.T

print(f"\nPrediction for user_0, item_0: {predictions[0, 0]:.2f}")
print(f"Actual rating: {R[0, 0]:.2f}")

# ============================================================================
# 3. NMF (NON-NEGATIVE MATRIX FACTORIZATION)
# ============================================================================

print("\n" + "=" * 70)
print("3. NMF (NON-NEGATIVE MATRIX FACTORIZATION)")
print("=" * 70)

print("""
NMF:
====

Constraint: All factors must be non-negative

R ≈ W × H

Where:
- W: (m × k) non-negative user factors
- H: (k × n) non-negative item factors

Advantages:
- More interpretable (no negative values)
- Parts-based representation
- Good for non-negative data (counts, ratings)

Disadvantages:
- Slower than SVD
- Non-convex (may not converge)
""")

# Prepare data (ensure non-negative)
R_nmf = R_sparse.copy()
R_nmf[R_nmf < 1] = 1  # Replace 0 with 1 for NMF

# Fit NMF
n_components = 5
nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=200)
W = nmf.fit_transform(R_nmf)
H = nmf.components_

print(f"NMF Results:")
print(f"  W shape: {W.shape}")
print(f"  H shape: {H.shape}")
print(f"  Reconstruction error: {nmf.reconstruction_err_:.3f}")

# Predict
R_nmf_pred = W @ H

print(f"\nNMF Prediction for user_0, item_0: {R_nmf_pred[0, 0]:.2f}")

# ============================================================================
# 4. FUNKSVD (GRADIENT DESCENT)
# ============================================================================

print("\n" + "=" * 70)
print("4. FUNKSVD (SGD-BASED)")
print("=" * 70)

class FunkSVD:
    """
    FunkSVD: Matrix Factorization using Stochastic Gradient Descent.

    Predicts: r_ui = μ + b_u + b_i + p_u · q_i
    """

    def __init__(self, n_factors=10, learning_rate=0.005, regularization=0.02, n_epochs=50):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs

    def fit(self, R):
        """
        Train the model.

        Parameters:
        -----------
        R : 2D array
            User-item matrix (sparse, 0 = not rated)
        """
        self.n_users, self.n_items = R.shape

        # Initialize
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))  # User factors
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))  # Item factors
        self.b_u = np.zeros(self.n_users)  # User biases
        self.b_i = np.zeros(self.n_items)  # Item biases
        self.mu = R[R > 0].mean()  # Global mean

        # Training
        for epoch in range(self.n_epochs):
            # Sample random ratings
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if R[u, i] > 0:
                        # Prediction
                        pred = self.predict_one(u, i)
                        error = R[u, i] - pred

                        # Update biases
                        self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                        self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])

                        # Update factors
                        P_u = self.P[u, :].copy()
                        self.P[u, :] += self.lr * (error * self.Q[i, :] - self.reg * self.P[u, :])
                        self.Q[i, :] += self.lr * (error * P_u - self.reg * self.Q[i, :])

            if (epoch + 1) % 10 == 0:
                train_rmse = self.compute_rmse(R)
                print(f"  Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}")

    def predict_one(self, u, i):
        """Predict rating for user-item pair."""
        return self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])

    def predict(self, R):
        """Predict all ratings."""
        return self.mu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :] + self.P @ self.Q.T

    def compute_rmse(self, R):
        """Compute RMSE on known ratings."""
        errors = []
        for u in range(self.n_users):
            for i in range(self.n_items):
                if R[u, i] > 0:
                    pred = self.predict_one(u, i)
                    errors.append((R[u, i] - pred) ** 2)
        return np.sqrt(np.mean(errors))

# Train FunkSVD
print("\nTraining FunkSVD...")
funk = FunkSVD(n_factors=5, learning_rate=0.01, regularization=0.02, n_epochs=20)
funk.fit(R_sparse)

# Evaluate
R_funksvd_pred = funk.predict(R_sparse)
print(f"\nFunkSVD Test RMSE: {np.sqrt(mean_squared_error(R[R>0], R_funksvd_pred[R>0])):.3f}")

# ============================================================================
# 5. ALS (ALTERNATING LEAST SQUARES)
# ============================================================================

print("\n" + "=" * 70)
print("5. ALS (ALTERNATING LEAST SQUARES)")
print("=" * 70)

class ALS:
    """
    Alternating Least Squares for implicit feedback.
    """

    def __init__(self, n_factors=10, regularization=0.1, n_iterations=10):
        self.n_factors = n_factors
        self.reg = regularization
        self.n_iterations = n_iterations

    def fit(self, R):
        """Train the model."""
        self.n_users, self.n_items = R.shape

        # Initialize
        np.random.seed(42)
        self.P = np.random.rand(self.n_users, self.n_factors) * 0.1
        self.Q = np.random.rand(self.n_items, self.n_factors) * 0.1

        # Get known ratings
        known = np.where(R > 0)

        for iteration in range(self.n_iterations):
            # Fix Q, solve for P
            for u in range(self.n_users):
                items_u = known[1][known[0] == u]
                if len(items_u) > 0:
                    Q_u = self.Q[items_u, :]
                    R_u = R[u, items_u]
                    A = Q_u.T @ Q_u + self.reg * np.eye(self.n_factors)
                    b = Q_u.T @ R_u
                    self.P[u, :] = np.linalg.solve(A, b)

            # Fix P, solve for Q
            for i in range(self.n_items):
                users_i = known[0][known[1] == i]
                if len(users_i) > 0:
                    P_i = self.P[users_i, :]
                    R_i = R[users_i, i]
                    A = P_i.T @ P_i + self.reg * np.eye(self.n_factors)
                    b = P_i.T @ R_i
                    self.Q[i, :] = np.linalg.solve(A, b)

            if (iteration + 1) % 5 == 0:
                pred = self.P @ self.Q.T
                rmse = np.sqrt(mean_squared_error(R[R>0], pred[R>0]))
                print(f"  Iteration {iteration+1}: RMSE = {rmse:.4f}")

    def predict(self):
        """Predict ratings."""
        return self.P @ self.Q.T

# Train ALS
print("\nTraining ALS...")
als = ALS(n_factors=5, regularization=0.1, n_iterations=15)
als.fit(R_sparse)

# Evaluate
R_als_pred = als.predict()
print(f"\nALS Test RMSE: {np.sqrt(mean_squared_error(R[R>0], R_als_pred[R>0])):.3f}")

# ============================================================================
# 6. BIASED MATRIX FACTORIZATION
# ============================================================================

print("\n" + "=" * 70)
print("6. BIASED MATRIX FACTORIZATION")
print("=" * 70)

print("""
Biased Matrix Factorization:
===========================

Prediction: r̂_ui = μ + b_u + b_i + p_u · q_i

Where:
- μ: Global average rating
- b_u: User bias (user tends to rate high/low)
- b_i: Item bias (some items are generally better)
- p_u: User latent factors
- q_i: Item latent factors

This accounts for:
- Some users are generous (positive bias)
- Some items are popular (positive bias)
""")

# Calculate biases
global_mean = R[R > 0].mean()
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)

# User bias
for u in range(n_users):
    rated = R[u, R[u] > 0]
    if len(rated) > 0:
        user_bias[u] = rated.mean() - global_mean

# Item bias
for i in range(n_items):
    rated = R[R[:, i] > 0, i]
    if len(rated) > 0:
        item_bias[i] = rated.mean() - global_mean

print(f"Global mean: {global_mean:.2f}")
print(f"User bias range: [{user_bias.min():.2f}, {user_bias.max():.2f}]")
print(f"Item bias range: [{item_bias.min():.2f}, {item_bias.max():.2f}]")

# ============================================================================
# 7. IMPLEMENTING WITH LIBRARIES
# ============================================================================

print("\n" + "=" * 70)
print("7. USING SURPRISE LIBRARY")
print("=" * 70)

try:
    from surprise import Dataset, Reader, SVD, NMF
    from surprise.model_selection import cross_validate

    print("\n" + "-" * 50)
    print("7.1 Using Surprise Library")
    print("-" * 50)

    # Create sample data
    data = []
    for u in range(n_users):
        for i in range(n_items):
            if R_sparse[u, i] > 0:
                data.append((u, i, R_sparse[u, i]))

    # Create dataset
    df = pd.DataFrame(data, columns=['user', 'item', 'rating'])

    # Load into Surprise
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df, reader)

    # Build trainset
    trainset = dataset.build_full_trainset()

    # SVD
    print("\nTraining SVD...")
    svd_algo = SVD(n_factors=10, n_epochs=20, random_state=42)
    svd_algo.fit(trainset)

    # Predict
    pred = svd_algo.predict(0, 5)  # user 0, item 5
    print(f"Prediction for user 0, item 5: {pred.est:.2f}")

    # Cross-validation
    print("\nCross-validation...")
    cv_results = cross_validate(svd_algo, dataset, measures=['RMSE', 'MAE'], cv=3, verbose=False)
    print(f"  RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"  MAE: {cv_results['test_mae'].mean():.4f}")

except ImportError:
    print("\nSurprise not installed. Install with: pip install scikit-surprise")

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("8. MODEL COMPARISON")
print("=" * 70)

print("\n" + "-" * 50)
print("Matrix Factorization Methods Comparison")
print("-" * 50)

print("""
| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| SVD | Fast, well-understood | Sensitive to missing values | Dense data |
| Truncated SVD | Handles sparsity | Approximate | Large sparse |
| NMF | Interpretable | Non-negative only | Non-negative |
| FunkSVD | Handles sparsity | Slow | Implicit feedback |
| ALS | Parallelizable | Memory intensive | Large scale |
""")

# Calculate RMSE for each method
methods = {
    'SVD (k=10)': R_reconstructed,
    'Truncated SVD': R_pred,
    'NMF': R_nmf_pred,
    'FunkSVD': R_funksvd_pred,
    'ALS': R_als_pred
}

print("\nRMSE Comparison:")
for name, pred in methods.items():
    rmse = np.sqrt(mean_squared_error(R[R>0], pred[R>0]))
    print(f"  {name}: {rmse:.4f}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Matrix Factorization
   - Decompose R ≈ U × V^T
   - Learn latent factors for users and items

2. SVD
   - Classic method
   - Requires dense matrix (fill zeros)
   - Truncated SVD handles sparse

3. NMF
   - Non-negative constraints
   - More interpretable
   - Good for ratings

4. FunkSVD (SGD)
   - Handles sparse matrices
   - Regularization prevents overfitting
   - Most popular for competitions

5. ALS
   - Alternating optimization
   - Good for parallelization
   - Used by Spotify, Netflix

6. Bias Terms
   - User and item biases important
   - Global mean + biases + factors

7. Libraries
   - Surprise (Python)
   - Implicit (fast, efficient)
   - TensorFlow Recommenders

Next: Deep Learning for Recommendations (03_deep_learning_recs.py)
""")
