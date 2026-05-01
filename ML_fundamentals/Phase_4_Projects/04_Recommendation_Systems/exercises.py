"""
Recommendation Systems - Practice Exercises
====================================

Complete these exercises to solidify your recommendation system skills.
Solutions are provided at the bottom.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXERCISE 1: User-Item Matrix
# ============================================================================

print("=" * 70)
print("EXERCISE 1: User-Item Matrix")
print("=" * 70)

# 1.1 Create user-item rating matrix
# TODO: Create a 5x6 matrix with random ratings 0-5
np.random.seed(42)

ratings = None  # TODO: Create 5 users, 6 items

print(f"Matrix shape: {ratings.shape}")
print(f"Rating distribution:\n{ratings}")

# 1.2 Calculate sparsity
# TODO: Calculate sparsity (percentage of zeros)
sparsity = None  # TODO: Calculate

print(f"\nSparsity: {sparsity*100:.1f}%")

# ============================================================================
# EXERCISE 2: Similarity Metrics
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Similarity Metrics")
print("=" * 70)

# 2.1 Calculate cosine similarity between two users
# TODO: Implement cosine_similarity
def cosine_similarity(v1, v2):
    """Calculate cosine similarity."""
    # TODO: Implement
    return 0.0

# Test
user1 = np.array([5, 4, 0, 2, 0])
user2 = np.array([3, 0, 5, 4, 0])

sim = cosine_similarity(user1, user2)
print(f"Cosine similarity: {sim:.4f}")

# ============================================================================
# EXERCISE 3: User-based CF
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: User-based Collaborative Filtering")
print("=" * 70)

# 3.1 Implement user-based CF prediction
# TODO: Implement user_based_cf
def user_based_cf(ratings, target_user, target_item, k=3):
    """
    Predict rating using user-based CF.
    """
    # TODO: Find users who rated target_item, calculate similarities,
    #       get top-k, weighted average
    return 0.0

# Test
ratings = np.array([
    [5, 4, 3, 0, 0],
    [4, 0, 5, 2, 0],
    [0, 3, 4, 5, 0],
    [2, 0, 0, 4, 5],
    [0, 0, 3, 0, 4]
])

pred = user_based_cf(ratings, target_user=0, target_item=3, k=2)
print(f"Predicted rating for user 0, item 3: {pred:.2f}")

# ============================================================================
# EXERCISE 4: Item-based CF
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Item-based Collaborative Filtering")
print("=" * 70)

# 4.1 Implement item-based CF
# TODO: Implement item_based_cf
def item_based_cf(ratings, target_user, target_item, k=3):
    """
    Predict rating using item-based CF.
    """
    # TODO: Find items rated by target_user, calculate similarities,
    #       get top-k, weighted average
    return 0.0

pred = item_based_cf(ratings, target_user=0, target_item=3, k=2)
print(f"Predicted rating for user 0, item 3: {pred:.2f}")

# ============================================================================
# EXERCISE 5: Matrix Factorization
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Matrix Factorization")
print("=" * 70)

# 5.1 Implement basic matrix factorization
# TODO: Implement matrix_factorization
def matrix_factorization(R, n_factors=2, n_epochs=100, lr=0.01, reg=0.01):
    """
    Simple matrix factorization using SGD.

    R: rating matrix (users x items)
    n_factors: number of latent factors
    """
    # TODO: Initialize U, V, biases
    # TODO: SGD loop
    # Return: predicted matrix
    return np.zeros_like(R)

# Test
R = np.array([
    [5, 4, 0, 2],
    [4, 0, 5, 2],
    [0, 3, 4, 5],
    [2, 0, 0, 4]
])

# R_pred = matrix_factorization(R, n_factors=2, n_epochs=50)
# print(f"Predicted ratings:\n{R_pred}")

# ============================================================================
# EXERCISE 6: Evaluation Metrics
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Evaluation Metrics")
print("=" * 70)

# 6.1 Calculate RMSE
# TODO: Implement rmse
def rmse(y_true, y_pred):
    """Calculate RMSE."""
    # TODO: Implement
    return 0.0

# 6.2 Calculate MAE
# TODO: Implement mae
def mae(y_true, y_pred):
    """Calculate MAE."""
    # TODO: Implement
    return 0.0

# 6.3 Calculate Precision@K
# TODO: Implement precision_at_k
def precision_at_k(y_true, y_pred, k):
    """Calculate Precision@K."""
    # TODO: Implement
    return 0.0

# Test
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

# print(f"RMSE: {rmse(y_true, y_pred):.4f}")
# print(f"MAE: {mae(y_true, y_pred):.4f}")
# print(f"Precision@3: {precision_at_k(y_true, y_pred, 3):.4f}")

# ============================================================================
# EXERCISE 7: NCF Model
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Neural Collaborative Filtering")
print("=" * 70)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Embedding, Flatten, Dense, Multiply, Concatenate

    # 7.1 Build NCF model
    # TODO: Implement build_ncf
    def build_ncf(n_users, n_items, embed_dim=32):
        """
        Build NCF model.

        Returns: Keras model
        """
        # TODO: Implement
        # User input, item input
        # Embeddings
        # GMF (multiply embeddings)
        # MLP (concatenate -> dense)
        # Combine GMF + MLP -> output
        return None

    # model = build_ncf(100, 50, embed_dim=16)
    # model.summary()

except ImportError:
    print("TensorFlow required")

# ============================================================================
# EXERCISE 8: Cold Start Problem
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Cold Start Solutions")
print("=" * 70)

# 8.1 Solutions for cold start
# TODO: List solutions for:
# - New user
# - New item

cold_start_solutions = {
    'new_user': [],  # TODO: Add solutions
    'new_item': []   # TODO: Add solutions
}

print("Cold Start Solutions:")
print(f"  New user: {cold_start_solutions['new_user']}")
print(f"  New item: {cold_start_solutions['new_item']}")

# ============================================================================
# EXERCISE 9: NDCG Calculation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9: NDCG Calculation")
print("=" * 70)

# 9.1 Implement NDCG@K
# TODO: Implement ndcg_at_k
def ndcg_at_k(y_true, y_pred, k):
    """
    Calculate NDCG@K.
    """
    # TODO: Implement
    # Sort by predicted, calculate DCG
    # Calculate IDCG
    # Return DCG/IDCG
    return 0.0

# Test
y_true = np.array([3, 2, 3, 0, 1, 2])
y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

# print(f"NDCG@3: {ndcg_at_k(y_true, y_pred, 3):.4f}")

# ============================================================================
# EXERCISE 10: Complete Recommender Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 10: Complete Pipeline")
print("=" * 70)

def recommend_pipeline(ratings, target_user, n_recommendations=5):
    """
    Complete recommendation pipeline.

    Steps:
    1. Preprocess data
    2. Calculate similarities
    3. Make predictions
    4. Generate recommendations
    5. Return top-N

    Parameters:
    -----------
    ratings : 2D array
        User-item matrix
    target_user : int
        User ID
    n_recommendations : int
        Number of recommendations

    Returns:
    --------
    list of (item_id, predicted_rating)
    """
    # TODO: Implement complete pipeline

    return []

# Test
# recommendations = recommend_pipeline(ratings, target_user=0, n_recommendations=3)
# print(f"Recommendations: {recommendations}")

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("""
EXERCISE 1:
1.1: ratings = np.random.randint(0, 6, (5, 6))
1.2: sparsity = (ratings == 0).sum() / ratings.size

EXERCISE 2:
2.1: def cosine_similarity(v1, v2):
        mask = (v1 > 0) & (v2 > 0)
        if mask.sum() == 0: return 0
        v1_c, v2_c = v1[mask], v2[mask]
        return np.dot(v1_c, v2_c) / (np.linalg.norm(v1_c) * np.linalg.norm(v2_c) + 1e-8)

EXERCISE 3:
3.1: def user_based_cf(ratings, target_user, target_item, k=3):
        # Find users who rated target_item
        users_who_rated = np.where(ratings[:, target_item] > 0)[0]
        if len(users_who_rated) == 0: return 0

        # Calculate similarities
        sims = []
        for u in users_who_rated:
            if u != target_user:
                sim = cosine_similarity(ratings[target_user], ratings[u])
                sims.append((u, sim, ratings[u, target_item]))

        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]

        if not top_k: return 0
        return sum(s*r for _, s, r in top_k) / sum(abs(s) for _, s, _ in top_k)

EXERCISE 4:
4.1: Similar to user-based but find items rated by target_user instead

EXERCISE 5:
5.1: def matrix_factorization(R, n_factors=2, n_epochs=100, lr=0.01, reg=0.01):
        n_users, n_items = R.shape
        np.random.seed(42)
        P = np.random.rand(n_users, n_factors) * 0.1
        Q = np.random.rand(n_items, n_factors) * 0.1

        for epoch in range(n_epochs):
            for u in range(n_users):
                for i in range(n_items):
                    if R[u, i] > 0:
                        pred = np.dot(P[u], Q[i])
                        error = R[u, i] - pred
                        P[u] += lr * (error * Q[i] - reg * P[u])
                        Q[i] += lr * (error * P[u] - reg * Q[i])

        return P @ Q.T

EXERCISE 6:
6.1: rmse = np.sqrt(mean_squared_error(y_true, y_pred))
6.2: mae = mean_absolute_error(y_true, y_pred)
6.3: top_k = np.argsort(y_pred)[-k:]
      return np.sum(y_true[top_k]) / k

EXERCISE 7:
7.1: def build_ncf(n_users, n_items, embed_dim=32):
        user_in = Input(shape=(1,))
        item_in = Input(shape=(1,))

        user_emb = Embedding(n_users, embed_dim)(user_in)
        item_emb = Embedding(n_items, embed_dim)(item_emb)

        # GMF
        gmf = Multiply()([Flatten()(user_emb), Flatten()(item_emb)])

        # MLP
        mlp = Concatenate()([Flatten()(user_emb), Flatten()(item_emb)])
        mlp = Dense(64, activation='relu')(mlp)
        mlp = Dense(32, activation='relu')(mlp)

        combined = Concatenate()([gmf, mlp])
        out = Dense(1, activation='sigmoid')(combined)

        return Model([user_in, item_in], out)

EXERCISE 8:
8.1: new_user: ['Ask for preferences', 'Recommend popular items', 'Content-based']
      new_item: ['Use content features', 'A/B testing', 'Explore-exploit']

EXERCISE 9:
9.1: def ndcg_at_k(y_true, y_pred, k):
        order = np.argsort(y_pred)[::-1][:k]
        y_true_sorted = y_true[order]

        dcg = np.sum((2**y_true_sorted - 1) / np.log2(np.arange(2, k+2)))
        ideal = np.sort(y_true)[::-1][:k]
        idcg = np.sum((2**ideal - 1) / np.log2(np.arange(2, k+2)))

        return dcg / idcg if idcg > 0 else 0

EXERCISE 10:
10.1: Combine all steps: preprocess, predict unrated items, return top-N
""")
