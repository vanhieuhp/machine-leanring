"""
Recurrent Neural Networks (RNN) - Deep Dive
============================================

This module covers:
- RNN concepts
- Sequence data processing
- LSTM and GRU
- Time series prediction
- Text generation
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        SimpleRNN, LSTM, GRU, Dense,
        Embedding, Dropout, Bidirectional
    )
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    RNN_AVAILABLE = True
except ImportError:
    RNN_AVAILABLE = False
    print("TensorFlow not available")

# ============================================================================
# 1. RNN CONCEPT
# ============================================================================

print("=" * 70)
print("1. RNN CONCEPT")
print("=" * 70)

print("""
Recurrent Neural Networks (RNN):

Designed for sequential data

The Problem with Feedforward Networks:
- Can't handle variable-length sequences
- Don't share parameters across positions

RNN Solution:
- Has "memory" of previous inputs
- Processes sequence one element at a time
- Output depends on previous computations

The Recurrence:
h_t = f(W * h_{t-1} + U * x_t)

Where:
- h_t: hidden state at time t
- x_t: input at time t
- W, U: weight matrices
""")

# ============================================================================
# 2. RNN ARCHITECTURE
# ============================================================================

print("\n" + "=" * 70)
print("2. RNN ARCHITECTURE")
print("=" * 70)

print("""
RNN Types:

1. One-to-One:
   - Single input, single output
   - Standard NN

2. One-to-Many:
   - Single input, sequence output
   - Image captioning

3. Many-to-One:
   - Sequence input, single output
   - Sentiment classification

4. Many-to-Many:
   - Sequence to sequence
   - Machine translation

Common RNN Layers:
- SimpleRNN: Basic RNN
- LSTM: Long Short-Term Memory
- GRU: Gated Recurrent Unit
""")

# ============================================================================
# 3. LSTM AND GRU
# ============================================================================

print("\n" + "=" * 70)
print("3. LSTM AND GRU")
print("=" * 70)

print("""
The Problem: Vanishing Gradient
- Standard RNN can't learn long-term dependencies
- Gradient becomes too small during backprop

LSTM (Long Short-Term Memory):
- Gates control information flow
- Forget gate: What to discard
- Input gate: What to store
- Output gate: What to output

GRU (Gated Recurrent Unit):
- Simplified version of LSTM
- Update gate: What to keep
- Reset gate: How much to forget
- Fewer parameters, faster to train

When to use:
- LSTM: Complex sequences, when accuracy matters
- GRU: Faster training, similar performance
- SimpleRNN: Quick prototyping (rarely used)
""")

# ============================================================================
# 4. SIMPLE RNN EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("4. SIMPLE RNN FOR TIME SERIES")
print("=" * 70)

if RNN_AVAILABLE:
    # Generate time series data
    def generate_sequence(length=50):
        x = np.linspace(0, 10, length)
        y = np.sin(x) + np.random.normal(0, 0.1, length)
        return y

    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    # Generate data
    np.random.seed(42)
    data = generate_sequence(200)
    seq_length = 10

    X, y = create_sequences(data, seq_length)
    X = X.reshape(-1, seq_length, 1)
    y = y.reshape(-1, 1)

    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build Simple RNN
    model = Sequential([
        SimpleRNN(32, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)

    # Predict
    y_pred = model.predict(X_test, verbose=0)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Simple RNN Time Series Prediction')
    plt.legend()
    plt.show()

    print(f"Test MSE: {np.mean((y_test - y_pred)**2):.4f}")

# ============================================================================
# 5. LSTM FOR SEQUENCE CLASSIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("5. LSTM FOR SEQUENCE CLASSIFICATION")
print("=" * 70)

if RNN_AVAILABLE:
    # Create sample sequence data
    np.random.seed(42)

    # Generate sequences of different classes
    X_sequences = []
    y_labels = []

    for _ in range(500):
        # Class 0: increasing trend
        seq = np.cumsum(np.random.randn(20))
        X_sequences.append(seq)
        y_labels.append(0)

        # Class 1: decreasing trend
        seq = -np.cumsum(np.random.randn(20))
        X_sequences.append(seq)
        y_labels.append(1)

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_labels)

    # Normalize
    X_seq = (X_seq - X_seq.mean()) / X_seq.std()

    # Reshape
    X_seq = X_seq.reshape(-1, 20, 1)

    # Split
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_labels[:split], y_labels[split:]

    # Build LSTM
    lstm_model = Sequential([
        LSTM(64, input_shape=(20, 1)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    lstm_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate
    loss, acc = lstm_model.evaluate(X_test, y_test)
    print(f"LSTM Test Accuracy: {acc:.4f}")

# ============================================================================
# 6. BIDIRECTIONAL LSTM
# ============================================================================

print("\n" + "=" * 70)
print("6. BIDIRECTIONAL LSTM")
print("=" * 70)

print("""
Bidirectional LSTM:
- Process sequence in both directions
- Forward: Beginning → End
- Backward: End → Beginning
- Concatenate outputs

When to use:
- Text classification
- Named Entity Recognition
- When future context helps
""")

if RNN_AVAILABLE:
    # Simple text classification example
    # Using IMDB-like data (simulated)
    from tensorflow.keras.datasets import imdb

    # Load data (small subset for demo)
    (X_train_text, y_train_text), (X_test_text, y_test_text) = imdb.load_data(num_words=1000)

    # Pad sequences
    maxlen = 100
    X_train_pad = pad_sequences(X_train_text, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_text, maxlen=maxlen)

    print(f"Training samples: {len(X_train_pad)}")
    print(f"Sequence length: {maxlen}")

    # Build Bidirectional LSTM
    bi_model = Sequential([
        Embedding(1000, 32, input_length=maxlen),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    bi_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    bi_model.summary()

    # Train for 1 epoch (demo)
    bi_model.fit(
        X_train_pad[:2000], y_train_text[:2000],
        epochs=1,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

# ============================================================================
# 7. GRU EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("7. GRU EXAMPLE")
print("=" * 70)

if RNN_AVAILABLE:
    # GRU is faster, often similar performance to LSTM
    gru_model = Sequential([
        GRU(32, input_shape=(10, 1)),
        Dense(1)
    ])

    gru_model.compile(optimizer='adam', loss='mse')

    # Use same time series data
    gru_model.fit(X_train, y_train, epochs=30, verbose=0)

    y_pred_gru = gru_model.predict(X_test, verbose=0)

    print(f"GRU Test MSE: {np.mean((y_test - y_pred_gru)**2):.4f}")
    print(f"(Lower is better)")

# ============================================================================
# 8. STACKED RNN
# ============================================================================

print("\n" + "=" * 70)
print("8. STACKED RNN")
print("=" * 70)

print("""
Stacked RNN:
- Multiple RNN layers
- Each layer processes output of previous
- Can learn more complex patterns

Example:
- Layer 1: Extract basic features
- Layer 2: Extract higher-level features
- ...
""")

if RNN_AVAILABLE:
    # Stacked LSTM
    stacked_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(20, 1)),
        LSTM(32),
        Dense(1)
    ])

    stacked_model.summary()

# ============================================================================
# 9. TEXT GENERATION WITH RNN
# ============================================================================

print("\n" + "=" * 70)
print("9. TEXT GENERATION (BASIC)")
print("=" * 70)

print("""
Text Generation with RNN:
- Train on sequence of characters/words
- Predict next character/word
- Sample from probabilities
- Can generate new text
""")

# ============================================================================
# 10. WHEN TO USE WHICH RNN
# ============================================================================

print("\n" + "=" * 70)
print("10. CHOOSING RNN TYPE")
print("=" * 70)

print("""
| Type      | Speed | Memory | Performance | Use Case           |
|-----------|-------|--------|--------------|--------------------|
| SimpleRNN | Fast  | Low    | Poor         | Quick prototyping  |
| LSTM      | Slow  | High   | Excellent    | Complex sequences  |
| GRU       | Medium| Medium | Good         | General purpose    |

Best Practices:
1. Start with GRU (good balance)
2. Use Bidirectional for text
3. Stack layers for complex patterns
4. Consider attention (transformers) for very long sequences
""")

print("\n" + "=" * 70)
print("RNN SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. RNNs process sequential data
2. LSTM/GRU solve vanishing gradient
3. Bidirectional for text tasks
4. Stacked layers for complexity
5. Consider Transformers for state-of-the-art

Common Applications:
- Time series prediction
- Text classification
- Machine translation
- Speech recognition
- Image captioning
""")
