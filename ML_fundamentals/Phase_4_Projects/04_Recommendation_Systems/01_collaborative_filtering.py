"""
Recommendation Systems - Part 1: Collaborative Filtering
========================================================

This module covers:
- User-item matrix
- Similarity metrics
- User-based CF
- Item-based CF
- Memory-based and model-based approaches

Based on: Movie Recommendation (MovieLens)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. USER-ITEM MATRIX
# ============================================================================

print("=" * 70)
print("1. USER-ITEM MATRIX")
print("=" * 70)

# Create sample movie ratings data
np.random.seed(42)

n_users = 10
n_items = 15

# User-item ratings matrix (users x items)
# 0 = not rated, 1-5 = ratings
ratings = np.zeros((n_users, n_items))

# Generate realistic ratings
for user in range(n_users):
    # Each user rates 5-10 movies
    n_ratings = np.random.randint(5, 11)
    items = np.random.choice(n_items, n_ratings, replace=False)
    for item in items:
        ratings[user, item] = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])

# Create DataFrame
user_ids = [f'user_{i}' for i in range(n_users)]
item_ids = [f'item_{i}' for i in range(n_items)]

df_ratings = pd.DataFrame(ratings, index=user_ids, columns=item_ids)

print(f"User-Item Matrix Shape: {df_ratings.shape}")
print(f"Users: {n_users}, Items: {n_items}")
print(f"\nSample ratings matrix:")
print(df_ratings.iloc[:5, :8])

# Statistics
total_ratings = np.sum(ratings > 0)
sparsity = 1 - (total_ratings / (n_users * n_items))

print(f"\nMatrix Statistics:")
print(f"  Total ratings: {total_ratings}")
print(f"  Sparsity: {sparsity*100:.1f}%")
print(f"  Avg ratings per user: {total_ratings/n_users:.1f}")
print(f"  Avg ratings per item: {total_ratings/n_items:.1f}")

# ============================================================================
# 2. SIMILARITY METRICS
# ============================================================================

print("\n" + "=" * 70)
print("2. SIMILARITY METRICS")
print("=" * 70)

# -------------------------------------------------------------------------
# 2.1 Cosine Similarity
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.1 Cosine Similarity")
print("-" * 50)

def cosine_similarity_vectors(v1, v2):
    """Calculate cosine similarity between two vectors."""
    # Only use common items
    mask = (v1 > 0) & (v2 > 0)
    if mask.sum() == 0:
        return 0

    v1_common = v1[mask]
    v2_common = v2[mask]

    dot = np.dot(v1_common, v2_common)
    norm1 = np.linalg.norm(v1_common)
    norm2 = np.linalg.norm(v2_common)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)

# Calculate similarity between users
user_similarities = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        if i != j:
            user_similarities[i, j] = cosine_similarity_vectors(
                ratings[i], ratings[j]
            )

print("User-User Similarity Matrix (first 5 users):")
print(pd.DataFrame(
    user_similarities[:5, :5],
    index=user_ids[:5],
    columns=user_ids[:5]
).round(3))

# Most similar users for user_0
user_0_sims = list(enumerate(user_similarities[0]))
user_0_sims.sort(key=lambda x: x[1], reverse=True)

print(f"\nMost similar users to user_0:")
for user_idx, sim in user_0_sims[1:6]:
    print(f"  {user_ids[user_idx]}: {sim:.3f}")

# -------------------------------------------------------------------------
# 2.2 Pearson Correlation
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.2 Pearson Correlation")
print("-" * 50)

def pearson_correlation(v1, v2):
    """Calculate Pearson correlation between two vectors."""
    # Only use common items
    mask = (v1 > 0) & (v2 > 0)
    if mask.sum() < 2:
        return 0

    v1_common = v1[mask]
    v2_common = v2[mask]

    # Calculate means
    mean1 = np.mean(v1_common)
    mean2 = np.mean(v2_common)

    # Calculate correlation
    num = np.sum((v1_common - mean1) * (v2_common - mean2))
    denom = np.sqrt(np.sum((v1_common - mean1)**2)) * np.sqrt(np.sum((v2_common - mean2)**2))

    if denom == 0:
        return 0

    return num / denom

# Calculate Pearson correlation
user_pearson = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        if i != j:
            user_pearson[i, j] = pearson_correlation(ratings[i], ratings[j])

print("User-User Pearson Correlation (first 5 users):")
print(pd.DataFrame(
    user_pearson[:5, :5],
    index=user_ids[:5],
    columns=user_ids[:5]
).round(3))

# -------------------------------------------------------------------------
# 2.3 Item Similarity
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.3 Item-Item Similarity")
print("-" * 50)

# Transpose for item-based
item_similarities = np.zeros((n_items, n_items))

for i in range(n_items):
    for j in range(n_items):
        if i != j:
            item_similarities[i, j] = cosine_similarity_vectors(
                ratings[:, i], ratings[:, j]
            )

# Most similar items to item_0
item_0_sims = list(enumerate(item_similarities[0]))
item_0_sims.sort(key=lambda x: x[1], reverse=True)

print(f"Most similar items to item_0:")
for item_idx, sim in item_0_sims[1:6]:
    print(f"  {item_ids[item_idx]}: {sim:.3f}")

# ============================================================================
# 3. USER-BASED COLLABORATIVE FILTERING
# ============================================================================

print("\n" + "=" * 70)
print("3. USER-BASED COLLABORATIVE FILTERING")
print("=" * 70)

def user_based_cf(ratings, user_idx, item_idx, k=3):
    """
    Predict rating for user-item pair using user-based CF.

    Parameters:
    -----------
    ratings : 2D array
        User-item ratings matrix
    user_idx : int
        User index
    item_idx : int
        Item index
    k : int
        Number of neighbors

    Returns:
    --------
    float
        Predicted rating
    """
    # Get users who rated this item
    users_who_rated = np.where(ratings[:, item_idx] > 0)[0]

    if len(users_who_rated) == 0:
        return 0  # No ratings for this item

    # Calculate similarities with these users
    similarities = []
    for other_user in users_who_rated:
        if other_user != user_idx:
            sim = cosine_similarity_vectors(ratings[user_idx], ratings[other_user])
            similarities.append((other_user, sim, ratings[other_user, item_idx]))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take top k
    top_k = similarities[:k]

    if len(top_k) == 0 or sum(s for _, s, _ in top_k) == 0:
        return 0

    # Weighted average
    num = sum(sim * rating for _, sim, rating in top_k)
    denom = sum(abs(sim) for _, sim, _ in top_k)

    if denom == 0:
        return 0

    return num / denom

# Predict rating for user_0 on item_8
target_user = 0
target_item = 8

# Check if already rated
if ratings[target_user, target_item] > 0:
    print(f"user_0 already rated item_8: {ratings[target_user, target_item]}")
else:
    predicted = user_based_cf(ratings, target_user, target_item, k=3)
    print(f"Predicted rating for user_0 on item_8: {predicted:.2f}")

# Evaluate on all missing ratings
print("\nUser-based CF Evaluation:")

errors = []
for user in range(n_users):
    for item in range(n_items):
        if ratings[user, item] > 0:  # Has actual rating
            predicted = user_based_cf(ratings, user, item, k=3)
            if predicted > 0:
                errors.append(abs(ratings[user, item] - predicted))

if len(errors) > 0:
    mae = np.mean(errors)
    print(f"  MAE: {mae:.3f}")

# ============================================================================
# 4. ITEM-BASED COLLABORATIVE FILTERING
# ============================================================================

print("\n" + "=" * 70)
print("4. ITEM-BASED COLLABORATIVE FILTERING")
print("=" * 70)

def item_based_cf(ratings, user_idx, item_idx, k=3):
    """
    Predict rating for user-item pair using item-based CF.
    """
    # Get items this user has rated
    items_user_rated = np.where(ratings[user_idx] > 0)[0]

    if len(items_user_rated) == 0:
        return 0

    # Calculate similarities with items this user rated
    similarities = []
    for rated_item in items_user_rated:
        if rated_item != item_idx:
            sim = cosine_similarity_vectors(ratings[:, item_idx], ratings[:, rated_item])
            similarities.append((rated_item, sim, ratings[user_idx, rated_item]))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take top k
    top_k = similarities[:k]

    if len(top_k) == 0 or sum(s for _, s, _ in top_k) == 0:
        return 0

    # Weighted average
    num = sum(sim * rating for _, sim, rating in top_k)
    denom = sum(abs(sim) for _, sim, _ in top_k)

    if denom == 0:
        return 0

    return num / denom

# Predict rating
predicted_item = item_based_cf(ratings, target_user, target_item, k=3)
print(f"Predicted rating (item-based) for user_0 on item_8: {predicted_item:.2f}")

# Evaluate
print("\nItem-based CF Evaluation:")

errors = []
for user in range(n_users):
    for item in range(n_items):
        if ratings[user, item] > 0:
            predicted = item_based_cf(ratings, user, item, k=3)
            if predicted > 0:
                errors.append(abs(ratings[user, item] - predicted))

if len(errors) > 0:
    mae = np.mean(errors)
    print(f"  MAE: {mae:.3f}")

# ============================================================================
# 5. MATRIX-BASED APPROACH (USING SKLEARN)
# ============================================================================

print("\n" + "=" * 70)
print("5. MATRIX-BASED COLLABORATIVE FILTERING")
print("=" * 70)

# Use sklearn's NearestNeighbors for efficient CF

from sklearn.neighbors import NearestNeighbors

# Create sparse matrix
sparse_ratings = csr_matrix(ratings)

# User-based KNN
print("\n" + "-" * 50)
print("5.1 User-based KNN")
print("-" * 50)

user_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=4)
user_knn.fit(sparse_ratings)

# Find similar users to user_0
distances, indices = user_knn.kneighbors(sparse_ratings[0], n_neighbors=4)

print(f"Users similar to user_0:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if i > 0:  # Skip self
        similarity = 1 - dist
        print(f"  {user_ids[idx]}: similarity = {similarity:.3f}")

# Item-based KNN
print("\n" + "-" * 50)
print("5.2 Item-based KNN")
print("-" * 50)

# Transpose for item-based
item_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=4)
item_knn.fit(sparse_ratings.T)

# Find similar items to item_0
distances, indices = item_knn.kneighbors(sparse_ratings[:, 0].T, n_neighbors=4)

print(f"Items similar to item_0:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if i > 0:  # Skip self
        similarity = 1 - dist
        print(f"  {item_ids[idx]}: similarity = {similarity:.3f}")

# ============================================================================
# 6. MAKING RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. GENERATING RECOMMENDATIONS")
print("=" * 70)

def get_user_recommendations(ratings, user_idx, n_recommendations=5):
    """
    Get top-N recommendations for a user.
    """
    # Items not yet rated
    unrated_items = np.where(ratings[user_idx] == 0)[0]

    if len(unrated_items) == 0:
        return []

    # Predict ratings for unrated items
    predictions = []
    for item_idx in unrated_items:
        pred = user_based_cf(ratings, user_idx, item_idx, k=3)
        predictions.append((item_idx, pred))

    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n_recommendations]

# Get recommendations for user_0
recommendations = get_user_recommendations(ratings, 0, n_recommendations=5)

print(f"\nTop 5 recommendations for user_0:")
for item_idx, pred_rating in recommendations:
    print(f"  {item_ids[item_idx]}: predicted rating = {pred_rating:.2f}")

# ============================================================================
# 7. COLD START PROBLEM
# ============================================================================

print("\n" + "=" * 70)
print("7. COLD START PROBLEM")
print("=" * 70)

print("""
Cold Start Problem:
=================

1. New User Problem:
   - No historical data
   - Cannot find similar users

   Solutions:
   - Ask for initial preferences
   - Recommend popular items
   - Use content-based features

2. New Item Problem:
   - No ratings for new items
   - Cannot find similar items

   Solutions:
   - Use item content features
   - A/B testing
   - Explore-exploit

3. Solutions Summary:
   - Hybrid approaches (CF + Content-based)
   - Matrix factorization
   - Knowledge-based recommendations
""")

# ============================================================================
# 8. EVALUATION METRICS
# ============================================================================

print("\n" + "=" * 70)
print("8. EVALUATION METRICS")
print("=" * 70)

def evaluate_recommender(ratings, train_ratio=0.8):
    """
    Evaluate recommender system using train-test split.
    """
    # Split ratings
    train = ratings.copy()
    test = ratings.copy()

    # Remove some ratings for testing
    test_indices = []
    for user in range(len(ratings)):
        rated_items = np.where(ratings[user] > 0)[0]
        if len(rated_items) > 0:
            n_test = max(1, int(len(rated_items) * (1 - train_ratio)))
            test_items = np.random.choice(rated_items, n_test, replace=False)
            test_indices.extend([(user, item) for item in test_items])

            for item in test_items:
                train[user, item] = 0

    # Evaluate
    errors = []
    for user, item in test_indices:
        if ratings[user, item] > 0:
            pred = user_based_cf(train, user, item, k=3)
            if pred > 0:
                errors.append(abs(ratings[user, item] - pred))

    if len(errors) > 0:
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        return mae, rmse

    return 0, 0

mae, rmse = evaluate_recommender(ratings)
print(f"\nEvaluation Results:")
print(f"  MAE: {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. User-Item Matrix
   - Rows = users, Columns = items
   - Sparse matrix (most cells are empty)
   - 0 = not rated, 1-5 = ratings

2. Similarity Metrics
   - Cosine: Direction of ratings
   - Pearson: Linear correlation
   - Jaccard: Binary ratings

3. User-based CF
   - Find similar users
   - Predict based on similar users' ratings
   - Good when users are similar

4. Item-based CF
   - Find similar items
   - Predict based on similar items user rated
   - More stable than user-based

5. Memory vs Model-based
   - Memory: Use all data (slower)
   - Model: Pre-compute (faster)

6. Cold Start
   - New users/items = no data
   - Solutions: Hybrid, popular items, content features

Next: Matrix Factorization (02_matrix_factorization.py)
""")
