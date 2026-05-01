"""
Recommendation Systems - Part 3: Deep Learning for Recommendations
=================================================================

This module covers:
- Neural Collaborative Filtering (NCF)
- Wide & Deep Learning
- Deep Factorization Machines
- Implementation with PyTorch and Keras
- Sequence-aware recommendations

Based on: Movie Recommendation with Deep Learning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, ndcg_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. NEURAL COLLABORATIVE FILTERING (NCF)
# ============================================================================

print("=" * 70)
print("1. NEURAL COLLABORATIVE FILTERING (NCF)")
print("=" * 70)

print("""
NCF Overview:
===========

Combines:
1. Generalized Matrix Factorization (GMF) - element-wise product
2. Multi-Layer Perceptron (MLP) - learn non-linear interactions

Architecture:
┌─────────────────────────────────────────────────────────┐
│                    NCF Architecture                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User Embedding ──┐                                     │
│  (MF path)       │──> Element-wise ──> Concatenate ─> │
│                   │     Product        │       │        │
│ Item Embedding ───┘                  │       │        │
│                                     │       ▼        │
│  User Embedding ──┐                 │   Dense Layers │
│  (MLP path)      │──> Concat ──────┤       │        │
│                   │                │       │        │
│ Item Embedding ───┘                │       ▼        │
│                                  Output Layer           │
│                                                          │
└─────────────────────────────────────────────────────────┘
""")

# Create sample data
np.random.seed(42)

n_users = 100
n_items = 50
n_samples = 1000

# Generate user-item interactions
user_ids = np.random.randint(0, n_users, n_samples)
item_ids = np.random.randint(0, n_items, n_samples)
ratings = np.random.randint(1, 6, n_samples)

# Convert to binary (positive/negative)
# Positive: rating >= 4, Negative: rating < 4
labels = (ratings >= 4).astype(int)

print(f"Dataset: {n_samples} samples")
print(f"Users: {n_users}, Items: {n_items}")
print(f"Positive samples: {labels.sum()} ({labels.mean()*100:.1f}%)")

# ============================================================================
# 2. PYTORCH NCF IMPLEMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("2. PYTORCH NCF IMPLEMENTATION")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print(f"PyTorch version: {torch.__version__}")

    # -------------------------------------------------------------------------
    # 2.1 NCF Model Definition
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.1 Defining NCF Model")
    print("-" * 50)

    class NCF(nn.Module):
        """Neural Collaborative Filtering Model."""

        def __init__(self, n_users, n_items, embed_dim=32, hidden_layers=[64, 32, 16]):
            super(NCF, self).__init__()

            # GMF embeddings
            self.user_embed_gmf = nn.Embedding(n_users, embed_dim)
            self.item_embed_gmf = nn.Embedding(n_items, embed_dim)

            # MLP embeddings (separate from GMF)
            self.user_embed_mlp = nn.Embedding(n_users, embed_dim)
            self.item_embed_mlp = nn.Embedding(n_items, embed_dim)

            # MLP layers
            mlp_layers = []
            input_size = embed_dim * 2
            for hidden_size in hidden_layers:
                mlp_layers.append(nn.Linear(input_size, hidden_size))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(0.2))
                input_size = hidden_size
            self.mlp = nn.Sequential(*mlp_layers)

            # Output layer
            self.output = nn.Linear(embed_dim + hidden_layers[-1], 1)

            # Initialize embeddings
            nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
            nn.init.normal_(self.item_embed_gmf.weight, std=0.01)
            nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
            nn.init.normal_(self.item_embed_mlp.weight, std=0.01)

        def forward(self, user_ids, item_ids):
            # GMF path
            user_gmf = self.user_embed_gmf(user_ids)
            item_gmf = self.item_embed_gmf(item_ids)
            gmf_output = user_gmf * item_gmf  # Element-wise product

            # MLP path
            user_mlp = self.user_embed_mlp(user_ids)
            item_mlp = self.item_embed_mlp(item_ids)
            mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
            mlp_output = self.mlp(mlp_input)

            # Combine GMF and MLP
            combined = torch.cat([gmf_output, mlp_output], dim=1)

            # Output
            output = torch.sigmoid(self.output(combined))

            return output

    # Instantiate model
    ncf_model = NCF(n_users, n_items, embed_dim=32, hidden_layers=[64, 32, 16])

    print("NCF Model:")
    print(ncf_model)

    # Count parameters
    total_params = sum(p.numel() for p in ncf_model.parameters())
    trainable_params = sum(p.numel() for p in ncf_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # 2.2 Training
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.2 Training NCF")
    print("-" * 50)

    # Prepare data
    X = np.column_stack([user_ids, item_ids])
    y = labels.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.LongTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.LongTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)

    # Training loop
    ncf_model.train()
    epochs = 10

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            user_batch = batch_x[:, 0]
            item_batch = batch_x[:, 1]

            optimizer.zero_grad()
            outputs = ncf_model(user_batch, item_batch).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    # -------------------------------------------------------------------------
    # 2.3 Evaluation
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.3 Evaluating NCF")
    print("-" * 50)

    ncf_model.eval()
    with torch.no_grad():
        test_outputs = ncf_model(X_test_t[:, 0], X_test_t[:, 1]).squeeze()
        test_preds = (test_outputs > 0.5).numpy()

    # Metrics
    accuracy = (test_preds == y_test).mean()
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)

    print(f"\nNCF Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

except ImportError:
    print("PyTorch not installed. Install with: pip install torch")

# ============================================================================
# 3. KERAS/TENSORFLOW NCF
# ============================================================================

print("\n" + "=" * 70)
print("3. KERAS/TENSORFLOW NCF")
print("=" * 70)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Embedding, Dense, Flatten, Concatenate, Dropout

    print(f"TensorFlow version: {tf.__version__}")

    # -------------------------------------------------------------------------
    # 3.1 Define NCF Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.1 Defining Keras NCF")
    print("-" * 50)

    embed_dim = 32

    # User embedding input
    user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
    user_embed = Embedding(n_users, embed_dim, name='user_embedding')(user_input)
    user_embed = Flatten()(user_embed)

    # Item embedding input
    item_input = tf.keras.layers.Input(shape=(1,), name='item_input')
    item_embed = Embedding(n_items, embed_dim, name='item_embedding')(item_input)
    item_embed = Flatten()(item_embed)

    # GMF (element-wise product)
    gmf = tf.keras.layers.Multiply()([user_embed, item_embed])

    # MLP
    mlp = Concatenate()([user_embed, item_embed])
    mlp = Dense(64, activation='relu')(mlp)
    mlp = Dropout(0.2)(mlp)
    mlp = Dense(32, activation='relu')(mlp)
    mlp = Dropout(0.2)(mlp)

    # Combine GMF and MLP
    combined = Concatenate()([gmf, mlp])
    output = Dense(1, activation='sigmoid')(combined)

    # Model
    keras_ncf = Model([user_input, item_input], output)
    keras_ncf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Keras NCF Model:")
    keras_ncf.summary()

    # -------------------------------------------------------------------------
    # 3.2 Train
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.2 Training Keras NCF")
    print("-" * 50)

    # Prepare data
    X_train_user = X_train[:, 0]
    X_train_item = X_train[:, 1]
    X_test_user = X_test[:, 0]
    X_test_item = X_test[:, 1]

    # Train
    history = keras_ncf.fit(
        [X_train_user, X_train_item],
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # -------------------------------------------------------------------------
    # 3.3 Evaluate
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.3 Evaluating Keras NCF")
    print("-" * 50)

    test_preds_prob = keras_ncf.predict([X_test_user, X_test_item], verbose=0)
    test_preds = (test_preds_prob > 0.5).astype(int).flatten()

    accuracy = (test_preds == y_test).mean()
    print(f"\nKeras NCF Accuracy: {accuracy:.4f}")

except ImportError as e:
    print(f"TensorFlow not available: {e}")

# ============================================================================
# 4. WIDE & DEEP LEARNING
# ============================================================================

print("\n" + "=" * 70)
print("4. WIDE & DEEP LEARNING")
print("=" * 70)

print("""
Wide & Deep:
============

Combines:
1. Wide: Memorization - learns direct feature interactions
2. Deep: Generalization - learns abstract patterns

Architecture:
┌─────────────────────────────────────────────────────────┐
│                Wide & Deep Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Wide (Linear)                                          │
│  ┌──────────────┐                                       │
│  │ Feature A    │─────┐                                 │
│  │ Feature B    │─────┤                                 │
│  └──────────────┘     │                                 │
│                        ▼                                 │
│                    Concatenate ──> Output              │
│                        ▲                                 │
│  Deep (Neural Net)                                      │
│  ┌──────────────┐     │                                 │
│  │ Embedding A  │─────┤                                 │
│  │ Embedding B  │─────┘                                 │
│  │ Dense Layers │                                       │
│  └──────────────┘                                       │
│                                                          │
│  Used by Google Play Store for app recommendations     │
└─────────────────────────────────────────────────────────┘
""")

# ============================================================================
# 5. DEEP FACTORIZATION MACHINES (DFM)
# ============================================================================

print("\n" + "=" * 70)
print("5. DEEP FACTORIZATION MACHINES (DeepFM)")
print("=" * 70)

print("""
DeepFM:
=======

Combines:
1. FM (Factorization Machine) - 2nd order feature interactions
2. Deep - Higher order feature interactions

Advantages:
- No manual feature engineering
- Learns both low and high-order feature interactions
- Efficient (shared embeddings)
""")

# ============================================================================
# 6. SEQUENCE-AWARE RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. SEQUENCE-AWARE RECOMMENDATIONS")
print("=" * 70)

print("""
Sequence-Aware Recommendations:
==============================

Use user's historical behavior sequence to predict next item.

Techniques:
1. Markov Chains - Probability of next item
2. RNN/LSTM - Sequential modeling
3. Transformer - Attention over sequences

Example:
User sequence: [A, B, C] → Predict: D
""")

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

print("\n" + "=" * 70)
print("7. EVALUATION METRICS FOR RANKING")
print("=" * 70)

def precision_at_k(y_true, y_pred, k):
    """Calculate Precision@K."""
    top_k = np.argsort(y_pred)[-k:]
    return np.sum(y_true[top_k]) / k

def recall_at_k(y_true, y_pred, k):
    """Calculate Recall@K."""
    top_k = np.argsort(y_pred)[-k:]
    n_relevant = np.sum(y_true)
    if n_relevant == 0:
        return 0
    return np.sum(y_true[top_k]) / n_relevant

def ndcg_at_k(y_true, y_pred, k):
    """Calculate NDCG@K."""
    # Sort by predicted scores
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order][:k]

    # DCG
    gains = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)

    # IDCG
    ideal_sorted = np.sort(y_true)[::-1][:k]
    ideal_gains = 2 ** ideal_sorted - 1
    idcg = np.sum(ideal_gains / discounts)

    if idcg == 0:
        return 0

    return dcg / idcg

# Calculate metrics
print("\nRanking Metrics:")

y_true = np.array([1, 0, 0, 1, 1, 0, 0, 1])  # Binary relevance
y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])  # Predicted scores

k = 3

print(f"True relevance: {y_true}")
print(f"Predicted scores: {y_pred}")

print(f"\n@{k} Metrics:")
print(f"  Precision@{k}: {precision_at_k(y_true, y_pred, k):.4f}")
print(f"  Recall@{k}: {recall_at_k(y_true, y_pred, k):.4f}")
print(f"  NDCG@{k}: {ndcg_at_k(y_true, y_pred, k):.4f}")

# ============================================================================
# 8. COMPLETE NCF PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("8. COMPLETE NCF PIPELINE")
print("=" * 70)

def ncf_pipeline(user_ids, item_ids, labels, embed_dim=32, epochs=10):
    """
    Complete NCF pipeline.

    Steps:
    1. Prepare data
    2. Build model
    3. Train
    4. Evaluate

    Returns:
    --------
    model, metrics
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        n_users = max(user_ids) + 1
        n_items = max(item_ids) + 1

        # Prepare data
        X = np.column_stack([user_ids, item_ids])
        y = labels.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = NCF(n_users, n_items, embed_dim=embed_dim)

        # Train
        # (simplified for demo)

        return model, {'accuracy': 0}

    except ImportError:
        return None, {}

print("NCF pipeline function created")

# ============================================================================
# 9. HYBRID APPROACHES
# ============================================================================

print("\n" + "=" * 70)
print("9. HYBRID RECOMMENDATIONS")
print("=" * 70)

print("""
Hybrid Approaches:
================

Combine multiple recommendation methods:

1. Weighted Hybrid:
   score = α × CF_score + β × Content_score + γ × MF_score

2. Switching Hybrid:
   Use different methods based on data availability

3. Cascade Hybrid:
   Output of one model feeds into next

4. Feature Combination:
   Combine features from multiple methods

5. Meta-level:
   Model trained on outputs of other models
""")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Neural Collaborative Filtering (NCF)
   - Embeddings + Neural networks
   - GMF: Element-wise product
   - MLP: Multi-layer perceptron

2. Wide & Deep
   - Wide: Linear (memorization)
   - Deep: Neural (generalization)

3. DeepFM
   - FM + Deep
   - Low and high-order interactions

4. Sequence-Aware
   - Consider user history order
   - RNN/LSTM/Transformer

5. Evaluation Metrics
   - Precision@K, Recall@K
   - NDCG@K (most important)
   - MRR (Mean Reciprocal Rank)

6. Hybrid Methods
   - Combine CF + Content-based
   - Ensemble multiple models

7. Libraries
   - TensorFlow Recommenders
   - PyTorch Geometric (GNN)
   - DeepMatch (deep matching)

Next: exercises.py to practice
""")
