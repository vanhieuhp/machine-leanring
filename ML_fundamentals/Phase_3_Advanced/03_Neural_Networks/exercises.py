"""
=================================================================
NEURAL NETWORKS — EXERCISES
=================================================================
5 hands-on exercises with increasing difficulty.
=================================================================
Prerequisites: pip install tensorflow
=================================================================
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️ TensorFlow not installed. Exercises 3-5 require it.")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 1: sklearn MLP Classifier (⭐⭐)                     ║
# ╚═════════════════════════════════════════════════════════════════╝
print("=" * 65)
print("EXERCISE 1: Build MLP with sklearn")
print("=" * 65)
print("""
📝 Task:
  1. Load breast cancer dataset
  2. Scale features with StandardScaler
  3. Train MLPClassifier with architecture (128, 64, 32)
  4. Try different activations: relu, tanh, logistic
  5. Compare test accuracy for each activation

🎯 Expected: ReLU should give best results
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for act in ['relu', 'tanh', 'logistic']:
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation=act,
                        max_iter=500, random_state=42)
    mlp.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_s))
    print(f"  {act:>10s}: accuracy = {acc:.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 2: Effect of Architecture Depth (⭐⭐)               ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 2: How Deep Should Your Network Be?")
print("=" * 65)
print("""
📝 Task:
  1. Use iris dataset (scaled)
  2. Test architectures: (16,), (64,), (64,32), (128,64,32), (256,128,64,32)
  3. Compare train and test accuracy
  4. Which architecture gives best generalization (test acc)?

🎯 Think: Does deeper always mean better?
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

archs = [(16,), (64,), (64, 32), (128, 64, 32), (256, 128, 64, 32)]
print(f"  {'Architecture':<25s} {'Train':>8s} {'Test':>8s}")
print("  " + "-" * 43)
for arch in archs:
    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
    mlp.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, mlp.predict(X_train_s))
    te = accuracy_score(y_test, mlp.predict(X_test_s))
    print(f"  {str(arch):<25s} {tr:>8.4f} {te:>8.4f}")
print("\n  💡 For small datasets like Iris, simple networks work best!")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 3: Keras Binary Classification (⭐⭐⭐)               ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 3: Build a Keras Model with Regularization")
print("=" * 65)
print("""
📝 Task:
  1. Load breast cancer dataset, scale features
  2. Build Keras model:
     - Dense(128) + BatchNorm + ReLU + Dropout(0.3)
     - Dense(64) + BatchNorm + ReLU + Dropout(0.3)
     - Dense(1, sigmoid)
  3. Use EarlyStopping (patience=10)
  4. Train and print test accuracy

🎯 Expected: accuracy > 0.97
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
if HAS_TF:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = keras.Sequential([
        layers.Dense(128, input_shape=(X_train_s.shape[1],)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_s, y_train, epochs=100, batch_size=32,
              validation_split=0.2, callbacks=[es], verbose=0)
    _, acc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"  Test Accuracy: {acc:.4f}")
else:
    print("  ⚠️ TensorFlow required")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 4: CNN on Fashion-MNIST (⭐⭐⭐)                      ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 4: Build a CNN for Fashion-MNIST")
print("=" * 65)
print("""
📝 Task:
  1. Load Fashion-MNIST from keras.datasets
  2. Normalize to [0,1], reshape to (28,28,1)
  3. Build CNN:
     - Conv2D(32, 3x3) + ReLU + MaxPool
     - Conv2D(64, 3x3) + ReLU + MaxPool
     - Flatten + Dense(64) + Dropout(0.5) + Dense(10, softmax)
  4. Train for 10 epochs
  5. Print test accuracy

🎯 Expected: accuracy > 0.89
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
if HAS_TF:
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Fashion-MNIST Test Accuracy: {acc:.4f}")
else:
    print("  ⚠️ TensorFlow required")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 5: LSTM for Sequence Classification (⭐⭐⭐⭐)         ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 5: LSTM Sequence Classification")
print("=" * 65)
print("""
📝 Task:
  1. Generate 1000 random sequences (length=50)
     - Class 1: sequences whose sum > 0
     - Class 0: sequences whose sum <= 0
  2. Build LSTM model:
     - LSTM(64) + Dense(32, relu) + Dense(1, sigmoid)
  3. Train and evaluate
  4. Compare SimpleRNN vs LSTM vs GRU

🎯 Expected: accuracy > 0.90
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
if HAS_TF:
    # Generate data
    np.random.seed(42)
    n_samples, seq_len = 2000, 50
    X_seq = np.random.randn(n_samples, seq_len, 1) * 0.5
    y_seq = (X_seq.sum(axis=1).flatten() > 0).astype(float)

    X_s_train, X_s_test = X_seq[:1600], X_seq[1600:]
    y_s_train, y_s_test = y_seq[:1600], y_seq[1600:]

    rnn_types = {
        "SimpleRNN": layers.SimpleRNN(64),
        "LSTM": layers.LSTM(64),
        "GRU": layers.GRU(64),
    }

    print(f"  {'Model':<12s} {'Accuracy':>10s} {'Params':>10s}")
    print("  " + "-" * 34)

    for name, rnn_layer in rnn_types.items():
        model = keras.Sequential([
            rnn_layer,
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
        # Rebuild input shape
        model.build(input_shape=(None, seq_len, 1))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_s_train, y_s_train, epochs=15, batch_size=32,
                  validation_split=0.2, verbose=0)
        _, acc = model.evaluate(X_s_test, y_s_test, verbose=0)
        print(f"  {name:<12s} {acc:>10.4f} {model.count_params():>10,}")
else:
    print("  ⚠️ TensorFlow required")

print("\n✅ Neural Networks exercises complete! Move on to 04_NLP_Basics next.")
