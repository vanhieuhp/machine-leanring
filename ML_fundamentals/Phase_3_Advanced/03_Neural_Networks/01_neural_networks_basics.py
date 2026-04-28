"""
=================================================================
01 - NEURAL NETWORKS BASICS: Perceptron, MLP, Forward Pass
=================================================================
Topics:
  1. Single neuron / perceptron from scratch
  2. Multi-Layer Perceptron with sklearn
  3. Introduction to Keras/TensorFlow
  4. Activation functions comparison
  5. Loss functions
  6. First Keras model
=================================================================
Prerequisites: pip install tensorflow
=================================================================
"""

import numpy as np
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Section 1: Single Neuron from Scratch ─────────────────────────
print("=" * 65)
print("SECTION 1: Single Neuron (Perceptron) from Scratch")
print("=" * 65)

print("""
  A single neuron computes:
    z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  (linear combination)
    ŷ = σ(z)                              (activation function)
""")


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    """Derivative of sigmoid."""
    s = sigmoid(z)
    return s * (1 - s)


class SimpleNeuron:
    """A single neuron (perceptron) with sigmoid activation."""

    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

    def forward(self, X):
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def train(self, X, y, epochs=100):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            losses.append(loss)

            # Backward pass (gradients)
            error = y_pred - y
            dw = (X.T @ error) / len(y)
            db = np.mean(error)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return losses

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


# Demo: simple binary classification
np.random.seed(42)
X_demo = np.random.randn(200, 2)
y_demo = (X_demo[:, 0] + X_demo[:, 1] > 0).astype(float)

neuron = SimpleNeuron(n_features=2, learning_rate=0.1)
losses = neuron.train(X_demo, y_demo, epochs=200)

y_pred = neuron.predict(X_demo)
acc = accuracy_score(y_demo, y_pred)

print(f"\n  Single Neuron Results:")
print(f"    Initial Loss: {losses[0]:.4f}")
print(f"    Final Loss:   {losses[-1]:.4f}")
print(f"    Accuracy:     {acc:.4f}")
print(f"    Weights:      {neuron.weights}")
print(f"    Bias:         {neuron.bias:.4f}")

# ── Section 2: MLP with scikit-learn ──────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Multi-Layer Perceptron with sklearn")
print("=" * 65)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for neural networks!)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- 2.1 Simple MLP ---
print("\n📌 2.1 Simple MLP (1 hidden layer):")
mlp_simple = MLPClassifier(
    hidden_layer_sizes=(64,),     # 1 hidden layer with 64 neurons
    activation='relu',
    max_iter=500,
    random_state=42,
)
mlp_simple.fit(X_train_s, y_train)
print(f"  Architecture: {X_train_s.shape[1]} → 64 → 1")
print(f"  Train: {accuracy_score(y_train, mlp_simple.predict(X_train_s)):.4f}")
print(f"  Test:  {accuracy_score(y_test, mlp_simple.predict(X_test_s)):.4f}")
print(f"  Iterations: {mlp_simple.n_iter_}")

# --- 2.2 Deeper MLP ---
print("\n📌 2.2 Deeper MLP (3 hidden layers):")
mlp_deep = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
)
mlp_deep.fit(X_train_s, y_train)
print(f"  Architecture: {X_train_s.shape[1]} → 128 → 64 → 32 → 1")
print(f"  Train: {accuracy_score(y_train, mlp_deep.predict(X_train_s)):.4f}")
print(f"  Test:  {accuracy_score(y_test, mlp_deep.predict(X_test_s)):.4f}")

# --- 2.3 Effect of hidden layer sizes ---
print("\n📌 2.3 Effect of Architecture:")
architectures = [
    (16,),
    (64,),
    (128,),
    (64, 32),
    (128, 64),
    (128, 64, 32),
    (256, 128, 64),
]

print(f"  {'Architecture':<22s} {'Train':>8s} {'Test':>8s} {'Params':>8s}")
print("  " + "-" * 50)

for arch in architectures:
    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
    mlp.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, mlp.predict(X_train_s))
    test_acc = accuracy_score(y_test, mlp.predict(X_test_s))

    # Count parameters
    layers = [X_train_s.shape[1]] + list(arch) + [1]
    n_params = sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))

    print(f"  {str(arch):<22s} {train_acc:>8.4f} {test_acc:>8.4f} {n_params:>8d}")

# ── Section 3: Activation Functions ───────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Activation Functions Comparison")
print("=" * 65)

for activation in ['relu', 'tanh', 'logistic']:
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation=activation,
        max_iter=500,
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)
    test_acc = accuracy_score(y_test, mlp.predict(X_test_s))
    print(f"  {activation:>10s}: test accuracy = {test_acc:.4f}, iterations = {mlp.n_iter_}")

print("""
  📊 Activation Function Guide:
  ┌──────────┬───────────┬────────────────────────────────┐
  │ Function │ Range     │ When to Use                    │
  ├──────────┼───────────┼────────────────────────────────┤
  │ ReLU     │ [0, ∞)    │ Hidden layers (DEFAULT)        │
  │ Tanh     │ (-1, 1)   │ Hidden layers (centered)       │
  │ Sigmoid  │ (0, 1)    │ Binary output layer            │
  │ Softmax  │ (0, 1)    │ Multi-class output layer       │
  └──────────┴───────────┴────────────────────────────────┘
""")

# ── Section 4: Solvers (Optimizers) ───────────────────────────────
print("=" * 65)
print("SECTION 4: Solvers / Optimizers")
print("=" * 65)

for solver in ['sgd', 'adam', 'lbfgs']:
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        solver=solver,
        max_iter=500,
        random_state=42,
    )
    mlp.fit(X_train_s, y_train)
    test_acc = accuracy_score(y_test, mlp.predict(X_test_s))
    print(f"  {solver:>8s}: test accuracy = {test_acc:.4f}, iterations = {mlp.n_iter_}")

print("""
  📊 Solver Guide:
    adam  → Good default, works for most cases
    sgd   → Better generalization, but needs tuning
    lbfgs → Good for small datasets
""")

# ── Section 5: Introduction to Keras ──────────────────────────────
print("=" * 65)
print("SECTION 5: First Keras/TensorFlow Model")
print("=" * 65)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
    print(f"  ✅ TensorFlow {tf.__version__} imported")
except ImportError:
    HAS_TF = False
    print("  ❌ TensorFlow not installed. Run: pip install tensorflow")

if HAS_TF:
    # Build a simple model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_s.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    print(f"\n  Model Summary:")
    model.summary()

    # Train
    print("\n  Training...")
    history = model.fit(
        X_train_s, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    # Evaluate
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"\n  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")

    # Training history
    print(f"\n  Training History (selected epochs):")
    print(f"  {'Epoch':>8s} {'Loss':>8s} {'Acc':>8s} {'Val Loss':>10s} {'Val Acc':>10s}")
    print("  " + "-" * 48)
    for epoch in [0, 9, 24, 49]:
        if epoch < len(history.history['loss']):
            print(f"  {epoch+1:>8d} "
                  f"{history.history['loss'][epoch]:>8.4f} "
                  f"{history.history['accuracy'][epoch]:>8.4f} "
                  f"{history.history['val_loss'][epoch]:>10.4f} "
                  f"{history.history['val_accuracy'][epoch]:>10.4f}")

# ── Section 6: Multi-class with Keras ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Multi-class Classification (Iris)")
print("=" * 65)

iris = load_iris()
X_i, y_i = iris.data, iris.target
X_i_train, X_i_test, y_i_train, y_i_test = train_test_split(
    X_i, y_i, test_size=0.2, random_state=42
)
scaler_i = StandardScaler()
X_i_train_s = scaler_i.fit_transform(X_i_train)
X_i_test_s = scaler_i.transform(X_i_test)

# sklearn MLP for multi-class
mlp_multi = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42,
)
mlp_multi.fit(X_i_train_s, y_i_train)
y_pred = mlp_multi.predict(X_i_test_s)

print(f"\n  sklearn MLP Multi-class (Iris):")
print(f"  Accuracy: {accuracy_score(y_i_test, y_pred):.4f}")
print(f"\n{classification_report(y_i_test, y_pred, target_names=iris.target_names)}")

if HAS_TF:
    # Keras multi-class
    model_multi = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(4,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax'),  # 3 classes → softmax
    ])
    model_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_multi.fit(X_i_train_s, y_i_train, epochs=100, batch_size=16, verbose=0)
    _, keras_acc = model_multi.evaluate(X_i_test_s, y_i_test, verbose=0)
    print(f"  Keras MLP Accuracy: {keras_acc:.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. Neural networks learn through forward pass + backpropagation
  2. ALWAYS scale features for neural networks!
  3. ReLU is the default hidden layer activation
  4. sigmoid → binary output, softmax → multi-class output
  5. Adam optimizer is a good default
  6. sklearn MLPClassifier is great for quick experiments
  7. Keras/TensorFlow for more control and deep architectures

📊 Output Layer Activation Guide:
  Binary classification  → sigmoid + binary_crossentropy
  Multi-class            → softmax + sparse_categorical_crossentropy
  Regression             → linear (none) + mse

📚 Next: 02_deep_networks.py (Deep Networks, Regularization)
""")
