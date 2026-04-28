"""
=================================================================
04 - RNN/LSTM: Recurrent Networks for Sequential Data
=================================================================
Topics:
  1. Why RNNs? Sequential data problems
  2. Simple RNN
  3. LSTM (Long Short-Term Memory)
  4. GRU (Gated Recurrent Unit)
  5. Bidirectional RNN
  6. Sequence classification example
  7. Time series forecasting
=================================================================
Prerequisites: pip install tensorflow
=================================================================
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("❌ TensorFlow not installed. Run: pip install tensorflow")

# ── Section 1: Why RNNs? ─────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 1: Why RNNs? Understanding Sequential Data")
print("=" * 65)

print("""
  Sequential data has ORDER that matters:
    • Text:       "I love cats" ≠ "cats love I"
    • Time series: stock prices, weather, sensor data
    • Audio:      speech signals
    • DNA:        gene sequences

  Regular neural networks (MLP) treat inputs independently.
  RNNs have MEMORY — they remember what came before.

  RNN Architecture:
    x₁ → [RNN] → h₁ ─┐
                       ↓
    x₂ → [RNN] → h₂ ─┐   (same weights reused!)
                       ↓
    x₃ → [RNN] → h₃ → output

    hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + b)
    Hidden state hₜ carries "memory" of past inputs
""")

# ── Section 2: Simple RNN ────────────────────────────────────────
print("=" * 65)
print("SECTION 2: Simple RNN")
print("=" * 65)

if HAS_TF:
    # Generate synthetic sequential data
    # Task: classify sequence as "ascending" (1) or "descending" (0)
    def generate_sequences(n_samples=1000, seq_length=20):
        X = np.zeros((n_samples, seq_length, 1))
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if np.random.rand() > 0.5:
                # Ascending
                start = np.random.rand() * 5
                X[i, :, 0] = np.linspace(start, start + np.random.rand() * 5, seq_length)
                X[i, :, 0] += np.random.randn(seq_length) * 0.3
                y[i] = 1
            else:
                # Descending
                start = np.random.rand() * 5 + 5
                X[i, :, 0] = np.linspace(start, start - np.random.rand() * 5, seq_length)
                X[i, :, 0] += np.random.randn(seq_length) * 0.3
                y[i] = 0
        return X, y

    X_seq, y_seq = generate_sequences(2000, 20)
    X_s_train, X_s_test, y_s_train, y_s_test = (
        X_seq[:1600], X_seq[1600:], y_seq[:1600], y_seq[1600:]
    )

    print(f"\n  Sequence shape: {X_seq.shape}")
    print(f"  (samples, timesteps, features) = ({X_seq.shape[0]}, {X_seq.shape[1]}, {X_seq.shape[2]})")

    # Simple RNN
    model_rnn = keras.Sequential([
        layers.SimpleRNN(32, input_shape=(20, 1)),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_rnn.fit(X_s_train, y_s_train, epochs=20, batch_size=32,
                  validation_split=0.2, verbose=0)
    _, rnn_acc = model_rnn.evaluate(X_s_test, y_s_test, verbose=0)
    print(f"\n  Simple RNN Accuracy: {rnn_acc:.4f}")
    print(f"  Parameters: {model_rnn.count_params():,}")

    print("""
  ⚠️ Problem with Simple RNN:
    • Vanishing gradient: can't learn long-range dependencies
    • After ~10-20 steps, gradient → 0, network "forgets"
    • Solution: LSTM and GRU
    """)

# ── Section 3: LSTM ───────────────────────────────────────────────
print("=" * 65)
print("SECTION 3: LSTM (Long Short-Term Memory)")
print("=" * 65)

print("""
  LSTM solves vanishing gradient with GATES:

  ┌──────────────────────────────────────┐
  │ LSTM Cell                            │
  │                                      │
  │  Forget Gate: fₜ = σ(Wf·[hₜ₋₁,xₜ]) │  → what to FORGET
  │  Input Gate:  iₜ = σ(Wi·[hₜ₋₁,xₜ]) │  → what to ADD
  │  Output Gate: oₜ = σ(Wo·[hₜ₋₁,xₜ]) │  → what to OUTPUT
  │                                      │
  │  Cell state: cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ │  → long-term memory
  │  Hidden:     hₜ = oₜ ⊙ tanh(cₜ)     │  → short-term output
  └──────────────────────────────────────┘

  The cell state cₜ can carry information over MANY time steps
  (like a highway that gradients flow through easily)
""")

if HAS_TF:
    model_lstm = keras.Sequential([
        layers.LSTM(32, input_shape=(20, 1)),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm.fit(X_s_train, y_s_train, epochs=20, batch_size=32,
                   validation_split=0.2, verbose=0)
    _, lstm_acc = model_lstm.evaluate(X_s_test, y_s_test, verbose=0)
    print(f"\n  LSTM Accuracy: {lstm_acc:.4f}")
    print(f"  Parameters: {model_lstm.count_params():,}")
    print(f"  (LSTM has 4x more params than SimpleRNN — 3 gates + cell)")

# ── Section 4: GRU ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: GRU (Gated Recurrent Unit)")
print("=" * 65)

print("""
  GRU is a simplified version of LSTM (2 gates instead of 3):

  ┌──────────────────────────────────────┐
  │ GRU Cell                             │
  │                                      │
  │  Reset Gate:  rₜ = σ(Wr·[hₜ₋₁,xₜ]) │  → what to forget
  │  Update Gate: zₜ = σ(Wz·[hₜ₋₁,xₜ]) │  → what to update
  │                                      │
  │  hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ        │  → combined memory
  └──────────────────────────────────────┘

  GRU vs LSTM:
    • Fewer parameters → faster training
    • Similar performance in most cases
    • Good default when unsure
""")

if HAS_TF:
    model_gru = keras.Sequential([
        layers.GRU(32, input_shape=(20, 1)),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_gru.fit(X_s_train, y_s_train, epochs=20, batch_size=32,
                  validation_split=0.2, verbose=0)
    _, gru_acc = model_gru.evaluate(X_s_test, y_s_test, verbose=0)
    print(f"\n  GRU Accuracy: {gru_acc:.4f}")
    print(f"  Parameters: {model_gru.count_params():,}")

# ── Section 5: Bidirectional RNN ──────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Bidirectional RNN")
print("=" * 65)

print("""
  Bidirectional processes sequence in BOTH directions:

    Forward:   x₁ → x₂ → x₃ → x₄   → h_fwd
    Backward:  x₁ ← x₂ ← x₃ ← x₄   → h_bwd
    Output:    [h_fwd ; h_bwd]        → concatenated

  Use when entire sequence is available (not real-time):
    ✅ Text classification, NER
    ❌ Real-time speech recognition
""")

if HAS_TF:
    model_bidir = keras.Sequential([
        layers.Bidirectional(layers.LSTM(32), input_shape=(20, 1)),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_bidir.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_bidir.fit(X_s_train, y_s_train, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)
    _, bidir_acc = model_bidir.evaluate(X_s_test, y_s_test, verbose=0)
    print(f"\n  Bidirectional LSTM Accuracy: {bidir_acc:.4f}")
    print(f"  Parameters: {model_bidir.count_params():,}")

# ── Section 6: Comparison ────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Head-to-Head Comparison")
print("=" * 65)

if HAS_TF:
    print(f"\n  {'Model':<22s} {'Accuracy':>10s} {'Params':>10s}")
    print("  " + "-" * 44)
    results = [
        ("SimpleRNN", rnn_acc, model_rnn.count_params()),
        ("LSTM", lstm_acc, model_lstm.count_params()),
        ("GRU", gru_acc, model_gru.count_params()),
        ("Bidirectional LSTM", bidir_acc, model_bidir.count_params()),
    ]
    for name, acc, params in results:
        print(f"  {name:<22s} {acc:>10.4f} {params:>10,}")

# ── Section 7: Time Series Forecasting ────────────────────────────
print("\n" + "=" * 65)
print("SECTION 7: Time Series Forecasting Example")
print("=" * 65)

if HAS_TF:
    # Generate sin wave data
    t = np.linspace(0, 100, 2000)
    data_ts = np.sin(t) + np.random.randn(len(t)) * 0.1

    # Create sequences: use past 30 values to predict next value
    window_size = 30
    X_ts, y_ts = [], []
    for i in range(len(data_ts) - window_size):
        X_ts.append(data_ts[i:i + window_size])
        y_ts.append(data_ts[i + window_size])
    X_ts, y_ts = np.array(X_ts), np.array(y_ts)
    X_ts = X_ts.reshape(-1, window_size, 1)

    # Split
    split = int(len(X_ts) * 0.8)
    X_ts_train, X_ts_test = X_ts[:split], X_ts[split:]
    y_ts_train, y_ts_test = y_ts[:split], y_ts[split:]

    print(f"\n  Time series forecasting: predict sin(t+1) from sin(t-29:t)")
    print(f"  Train: {len(X_ts_train)}, Test: {len(X_ts_test)}")

    # LSTM for forecasting
    model_ts = keras.Sequential([
        layers.LSTM(50, input_shape=(window_size, 1)),
        layers.Dense(25, activation='relu'),
        layers.Dense(1),  # Linear output for regression
    ])
    model_ts.compile(optimizer='adam', loss='mse')
    model_ts.fit(X_ts_train, y_ts_train, epochs=20, batch_size=32,
                 validation_split=0.1, verbose=0)

    y_pred_ts = model_ts.predict(X_ts_test, verbose=0).flatten()
    mse = np.mean((y_ts_test - y_pred_ts) ** 2)
    mae = np.mean(np.abs(y_ts_test - y_pred_ts))

    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. RNNs process sequential data with temporal memory
  2. SimpleRNN is fast but suffers from vanishing gradient
  3. LSTM uses 3 gates to maintain long-term memory
  4. GRU is simpler (2 gates), similar performance to LSTM
  5. Bidirectional is better when full sequence is available
  6. Input shape: (batch, timesteps, features)

🔑 Quick Decision Guide:
  Short sequences (<50 steps)   → GRU or SimpleRNN
  Long sequences (>50 steps)    → LSTM
  Full sequence available       → Bidirectional LSTM
  Speed is priority             → GRU
  Modern NLP                    → Transformers (not RNN!)

📚 Next: exercises.py (Neural Network Practice Problems)
""")
