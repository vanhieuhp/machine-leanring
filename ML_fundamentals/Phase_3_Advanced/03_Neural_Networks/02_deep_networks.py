"""
=================================================================
02 - DEEP NETWORKS: Regularization, Optimizers, and Best Practices
=================================================================
Topics:
  1. Deeper architectures
  2. Dropout regularization
  3. Batch Normalization
  4. Learning rate scheduling
  5. Early stopping
  6. Weight initialization
  7. Practical deep learning workflow
=================================================================
Prerequisites: pip install tensorflow
=================================================================
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    HAS_TF = True
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("❌ TensorFlow not installed. Run: pip install tensorflow")

# ── Dataset ──────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=5000, n_features=50, n_informative=25,
    n_redundant=10, n_classes=2, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

if not HAS_TF:
    print("\n⚠️ This file requires TensorFlow. Install with: pip install tensorflow")
    print("   Showing concepts only.\n")

# ── Section 1: Overfitting Demo ───────────────────────────────────
print("=" * 65)
print("SECTION 1: Overfitting in Deep Networks")
print("=" * 65)

if HAS_TF:
    # Overly complex model (will overfit)
    model_overfit = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(50,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\n  Training overly complex model (no regularization)...")
    hist = model_overfit.fit(
        X_train_s, y_train, epochs=50, batch_size=32,
        validation_split=0.2, verbose=0,
    )
    train_acc = hist.history['accuracy'][-1]
    val_acc = hist.history['val_accuracy'][-1]
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    print(f"  Gap:            {train_acc - val_acc:.4f}")
    if train_acc - val_acc > 0.05:
        print("  ⚠️ Overfitting detected! Train >> Val")

# ── Section 2: Dropout ────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Dropout Regularization")
print("=" * 65)

print("""
  Dropout: randomly "drops" neurons during training
    • Each neuron has probability p of being set to 0
    • Forces the network to not rely on any single neuron
    • Acts like training an ensemble of sub-networks

    Training:  [●] [○] [●] [●] [○] [●]   (○ = dropped)
    Inference: [●] [●] [●] [●] [●] [●]   (all active, scaled)
""")

if HAS_TF:
    print("  Testing different dropout rates:\n")
    print(f"  {'Dropout':>10s} {'Train':>8s} {'Val':>8s} {'Gap':>8s}")
    print("  " + "-" * 38)

    for dropout_rate in [0.0, 0.2, 0.3, 0.5, 0.7]:
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(50,)),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train_s, y_train, epochs=30, batch_size=32,
                         validation_split=0.2, verbose=0)
        tr = hist.history['accuracy'][-1]
        va = hist.history['val_accuracy'][-1]
        print(f"  {dropout_rate:>10.1f} {tr:>8.4f} {va:>8.4f} {tr-va:>8.4f}")

    print("\n  💡 Dropout 0.2-0.5 usually works best")

# ── Section 3: Batch Normalization ────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Batch Normalization")
print("=" * 65)

print("""
  Batch Norm normalizes layer outputs to have mean≈0, std≈1
    • Speeds up training (can use higher learning rates)
    • Acts as mild regularization
    • Place AFTER Dense, BEFORE activation (or after — both work)
""")

if HAS_TF:
    # Without BatchNorm
    model_nobn = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(50,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_nobn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h1 = model_nobn.fit(X_train_s, y_train, epochs=30, batch_size=32,
                        validation_split=0.2, verbose=0)

    # With BatchNorm
    model_bn = keras.Sequential([
        layers.Dense(256, input_shape=(50,)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_bn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h2 = model_bn.fit(X_train_s, y_train, epochs=30, batch_size=32,
                      validation_split=0.2, verbose=0)

    print(f"\n  Without BatchNorm: val_acc = {h1.history['val_accuracy'][-1]:.4f}")
    print(f"  With BatchNorm:    val_acc = {h2.history['val_accuracy'][-1]:.4f}")

# ── Section 4: Early Stopping ─────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Early Stopping")
print("=" * 65)

print("""
  Stop training when validation loss stops improving:
    • patience: how many epochs to wait before stopping
    • restore_best_weights: go back to the best epoch
""")

if HAS_TF:
    model_es = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(50,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_es.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    hist_es = model_es.fit(
        X_train_s, y_train, epochs=200,  # set high, early stopping will handle it
        batch_size=32, validation_split=0.2,
        callbacks=[early_stop], verbose=0,
    )

    actual_epochs = len(hist_es.history['loss'])
    print(f"\n  Set epochs: 200, Actual epochs: {actual_epochs}")
    print(f"  Final val_accuracy: {hist_es.history['val_accuracy'][-1]:.4f}")
    print(f"  Test accuracy: {model_es.evaluate(X_test_s, y_test, verbose=0)[1]:.4f}")

# ── Section 5: Learning Rate Scheduling ───────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Learning Rate Scheduling")
print("=" * 65)

if HAS_TF:
    print("\n  Comparing learning rate strategies:\n")

    # Fixed learning rates
    results = {}
    for lr in [0.01, 0.001, 0.0001]:
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(50,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train_s, y_train, epochs=30, batch_size=32,
                         validation_split=0.2, verbose=0)
        results[f"Fixed lr={lr}"] = hist.history['val_accuracy'][-1]

    # ReduceLROnPlateau
    model_rlr = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(50,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_rlr.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                      loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    hist_rlr = model_rlr.fit(X_train_s, y_train, epochs=30, batch_size=32,
                             validation_split=0.2, callbacks=[reduce_lr], verbose=0)
    results["ReduceLROnPlateau"] = hist_rlr.history['val_accuracy'][-1]

    print(f"  {'Strategy':<25s} {'Val Accuracy':>12s}")
    print("  " + "-" * 40)
    for name, acc in results.items():
        print(f"  {name:<25s} {acc:>12.4f}")

# ── Section 6: L2 Regularization ─────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: L2 (Weight Decay) Regularization")
print("=" * 65)

if HAS_TF:
    # Without L2
    model_no_l2 = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(50,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_no_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h1 = model_no_l2.fit(X_train_s, y_train, epochs=30, batch_size=32,
                         validation_split=0.2, verbose=0)

    # With L2
    model_l2 = keras.Sequential([
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=(50,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(1, activation='sigmoid'),
    ])
    model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h2 = model_l2.fit(X_train_s, y_train, epochs=30, batch_size=32,
                      validation_split=0.2, verbose=0)

    print(f"\n  Without L2: train={h1.history['accuracy'][-1]:.4f}, val={h1.history['val_accuracy'][-1]:.4f}")
    print(f"  With L2:    train={h2.history['accuracy'][-1]:.4f}, val={h2.history['val_accuracy'][-1]:.4f}")

# ── Section 7: Best Practice Architecture ─────────────────────────
print("\n" + "=" * 65)
print("SECTION 7: Putting It All Together")
print("=" * 65)

if HAS_TF:
    print("\n  Building a well-regularized deep network:")

    model_best = keras.Sequential([
        # Layer 1
        layers.Dense(256, input_shape=(50,)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        # Layer 2
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        # Layer 3
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        # Output
        layers.Dense(1, activation='sigmoid'),
    ])

    model_best.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0),
    ]

    hist_best = model_best.fit(
        X_train_s, y_train, epochs=200, batch_size=32,
        validation_split=0.2, callbacks=my_callbacks, verbose=0,
    )

    test_loss, test_acc = model_best.evaluate(X_test_s, y_test, verbose=0)
    print(f"\n  Epochs trained:  {len(hist_best.history['loss'])}")
    print(f"  Final train acc: {hist_best.history['accuracy'][-1]:.4f}")
    print(f"  Final val acc:   {hist_best.history['val_accuracy'][-1]:.4f}")
    print(f"  Test accuracy:   {test_acc:.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. Use Dropout (0.2-0.5) to prevent overfitting
  2. Use Batch Normalization for faster, stable training
  3. Use Early Stopping to find optimal epoch count
  4. Use ReduceLROnPlateau for adaptive learning rates
  5. L2 regularization adds penalty for large weights
  6. Combine: BatchNorm + Dropout + EarlyStopping + LR scheduling

📋 Best Practice Architecture Template:
    Dense → BatchNorm → ReLU → Dropout → ... → Output

📚 Next: 03_cnn_images.py (CNNs for Image Classification)
""")
