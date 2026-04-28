# Recommendation Systems - Learning Guide

## What are Recommendation Systems?

Recommendation systems predict user preferences and suggest relevant items from a large collection.

## Why Recommendation Systems Matter

- **E-commerce**: Product recommendations (Amazon)
- **Streaming**: Movies/music (Netflix, Spotify)
- **Social Media**: Content recommendations (YouTube, TikTok)
- **Services**: Restaurant, travel, job recommendations

## Learning Objectives

By the end of this section, you'll master:

### Traditional Methods
1. **Collaborative Filtering** - User-user, Item-item
2. **Content-Based Filtering** - Item features
3. **Matrix Factorization** - SVD, NMF

### Modern Deep Learning
1. **Neural Collaborative Filtering** - NCF
2. **Wide & Deep Learning** - Google
3. **Transformers for Recommendations** - BERT4Rec

### Production Systems
1. **Hybrid Methods** - Combining approaches
2. **Evaluation Metrics** - Precision, Recall, NDCG
3. **Real-world Implementation** - Serving, A/B testing

## Key Concepts

### 1. Problem Formulation

```
┌─────────────────────────────────────────────────────────────┐
│            RECOMMENDATION PROBLEM                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Given:                                                      │
│  - Users: U = {u1, u2, ..., u_m}                           │
│  - Items: I = {i1, i2, ..., i_n}                           │
│  - Ratings: R = {r_ui} (explicit or implicit)              │
│                                                              │
│  Goal:                                                       │
│  - Predict rating for user-item pair (r_ui)                │
│  - Top-K recommendations for each user                     │
│                                                              │
│  Types:                                                      │
│  - Explicit: Star ratings, likes (1-5)                     │
│  - Implicit: Views, clicks, purchases (binary)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Collaborative Filtering

```
┌─────────────────────────────────────────────────────────────┐
│            COLLABORATIVE FILTERING                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User-User CF:                                              │
│  ┌─────────────────────────────────────────────┐           │
│  │  Find similar users                        │           │
│  │  Recommend what similar users liked        │           │
│  └─────────────────────────────────────────────┘           │
│                                                              │
│  Item-Item CF:                                              │
│  ┌─────────────────────────────────────────────┐           │
│  │  Find similar items                        │           │
│  │  Recommend similar to what user liked       │           │
│  └─────────────────────────────────────────────┘           │
│                                                              │
│  Similarity: Cosine, Pearson, Jaccard                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. Matrix Factorization

```
┌─────────────────────────────────────────────────────────────┐
│            MATRIX FACTORIZATION                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User-Item Matrix (R):                                      │
│  ┌───┬───┬───┬───┬───┐                                     │
│  │ 5 │ 3 │ ? │ 4 │ ? │   ← User preferences               │
│  ├───┼───┼───┼───┼───┤                                     │
│  │ 4 │ ? │ 2 │ 3 │ ? │                                     │
│  ├───┼───┼───┼───┼───┤                                     │
│  │ ? │ 5 │ 4 │ ? │ 5 │   Decompose: R ≈ U × V^T           │
│  ├───┼───┼───┼───┼───┤                                     │
│  │ 2 │ ? │ ? │ 5 │ 3 │   U: User embeddings (k dims)      │
│  └───┴───┴───┴───┴───┘   V: Item embeddings (k dims)      │
│                                                              │
│  Learn: U (m×k), V (n×k) to minimize ||R - UV^T||^2        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error |
| **MAE** | Mean Absolute Error |
| **Precision@K** | Relevant items in top-K |
| **Recall@K** | Relevant items found |
| **NDCG@K** | Normalized DCG (rank-aware) |
| **Hit Rate** | At least one relevant item in top-K |

### 5. Cold Start Problem

```
┌─────────────────────────────────────────────────────────────┐
│            COLD START PROBLEM                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Cold Start:                                            │
│  - New user with no history                                  │
│  - Solution: Ask for preferences, popular items             │
│                                                              │
│  Item Cold Start:                                             │
│  - New item not yet rated                                    │
│  - Solution: Content features, A/B testing                  │
│                                                              │
│  Solution:                                                   │
│  - Hybrid approach (CF + Content)                          │
│  - Explore-exploit balance                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Study Path

### Week 1: Collaborative Filtering
1. **Start with**: Basic CF
   - User-based CF
   - Item-based CF
   - Similarity metrics

2. **Then**: Matrix Factorization
   - SVD (Singular Value Decomposition)
   - NMF (Non-negative Matrix Factorization)
   - ALS (Alternating Least Squares)

### Week 2: Deep Learning for Recommendations
3. **Next**: Neural Collaborative Filtering
   - Embedding layers
   - NCF architecture
   - Wide & Deep

### Week 3: Advanced Topics
4. **Then**: Advanced Techniques
   - Sequence-aware recommendations
   - Context-aware recommendations
   - Reinforcement learning for recommendations

### Week 4: Project & Production
5. **Finally**: Complete Project
   - Movie recommendation system
   - Implement hybrid approach
   - Evaluate with multiple metrics

## Common Mistakes to Avoid

1. **Ignoring implicit feedback** - Views, clicks are valuable
2. **Not handling sparsity** - Most user-item pairs are unrated
3. **Cold start** - Plan for new users/items
4. **Overfitting** - Regularization is crucial
5. **Only optimizing for accuracy** - Diversity, novelty matter

## Popular Libraries

| Library | Description |
|---------|-------------|
| **Surprise** | Classic CF algorithms |
| **TensorFlow Recommenders** | Google's TF-based |
| **PyTorch Geometric** | Graph-based recs |
| **LensKit** | Research-focused |

---

**Difficulty**: Expert

**Estimated Time**: 1 month

**Prerequisites**: Phase 1 (NumPy, Pandas), Phase 2 (Neural Networks)

**Next**: Career in Machine Learning
