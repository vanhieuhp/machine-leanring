"""
Perceptron and Neural Network Fundamentals
==========================================

This module covers:
- Perceptron algorithm
- Activation functions
- Building neural networks from scratch
- Understanding forward and backward propagation
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. PERCEPTRON CONCEPT
# ============================================================================

print("=" * 70)
print("1. PERCEPTRON CONCEPT")
print("=" * 70)

print("""
Perceptron: The simplest neural network

Structure:
- Input: x₁, x₂, ..., xₙ
- Weights: w₁, w₂, ..., wₙ
- Bias: b
- Output: y = activation(w·x + b)

The perceptron makes a decision:
- If w·x + b > 0 → output 1
- Else → output 0

The perceptron can learn weights through training!
""")

# ============================================================================
# 2. PERCEPTRON FROM SCRATCH
# ============================================================================

print("\n" + "=" * 70)
print("2. PERCEPTRON FROM SCRATCH")
print("=" * 70)

class Perceptron:
    """Simple Perceptron implementation"""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def activation(self, x):
        """Step function"""
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        """Make predictions"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

    def fit(self, X, y):
        """Train the perceptron"""
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Forward pass
                linear = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear)

                # Update weights
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

# Test perceptron on AND problem
print("Training perceptron on AND problem:")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND logic

perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X, y)

print("Predictions:")
for x in X:
    print(f"{x} -> {perceptron.predict([x])[0]}")

# ============================================================================
# 3. ACTIVATION FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("3. ACTIVATION FUNCTIONS")
print("=" * 70)

print("""
Activation Functions: Add non-linearity

1. Step Function:
   f(x) = 1 if x >= 0, else 0
   - Used in original perceptron
   - Not differentiable at 0

2. Sigmoid:
   f(x) = 1 / (1 + e^(-x))
   - S-shaped curve
   - Output between 0 and 1
   - Good for probability

3. Tanh:
   f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Output between -1 and 1
   - Zero-centered

4. ReLU (Rectified Linear Unit):
   f(x) = max(0, x)
   - Most popular!
   - Fast to compute
   - Works well in practice

5. Softmax:
   f(x_i) = e^x_i / Σ e^x_j
   - For multi-class output
   - Outputs sum to 1
""")

# Visualize activation functions
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Step
axes[0, 0].plot(x, np.where(x >= 0, 1, 0), 'b-', linewidth=2)
axes[0, 0].set_title('Step Function')
axes[0, 0].grid(True, alpha=0.3)

# Sigmoid
axes[0, 1].plot(x, 1 / (1 + np.exp(-x)), 'b-', linewidth=2)
axes[0, 1].set_title('Sigmoid')
axes[0, 1].grid(True, alpha=0.3)

# Tanh
axes[0, 2].plot(x, np.tanh(x), 'b-', linewidth=2)
axes[0, 2].set_title('Tanh')
axes[0, 2].grid(True, alpha=0.3)

# ReLU
axes[1, 0].plot(x, np.maximum(0, x), 'b-', linewidth=2)
axes[1, 0].set_title('ReLU')
axes[1, 0].grid(True, alpha=0.3)

# Leaky ReLU
axes[1, 1].plot(x, np.where(x > 0, x, 0.01 * x), 'b-', linewidth=2)
axes[1, 1].set_title('Leaky ReLU')
axes[1, 1].grid(True, alpha=0.3)

# Softmax (for 3 classes)
x_softmax = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2]])
softmax_output = np.exp(x_softmax) / np.sum(np.exp(x_softmax), axis=1, keepdims=True)
axes[1, 2].bar(['Class 0', 'Class 1', 'Class 2'], softmax_output[0])
axes[1, 2].set_title('Softmax')
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# ============================================================================
# 4. MULTI-LAYER PERCEPTRON
# ============================================================================

print("\n" + "=" * 70)
print("4. MULTI-LAYER PERCEPTRON")
print("=" * 70)

print("""
MLP (Multi-Layer Perceptron):
- Input layer
- Hidden layer(s)
- Output layer

Why hidden layers?
- Can learn non-linear patterns
- Single perceptron can only learn linear decisions

Example: XOR Problem
- NOT linearly separable
- Requires hidden layer!
""")

# XOR problem
print("XOR Problem:")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR logic

print("Data:")
for x, y in zip(X_xor, y_xor):
    print(f"{x} -> {y}")

print("\nSingle perceptron CANNOT solve this!")
print("But MLP with hidden layer CAN!")

# ============================================================================
# 5. NEURAL NETWORK FROM SCRATCH
# ============================================================================

print("\n" + "=" * 70)
print("5. NEURAL NETWORK FROM SCRATCH")
print("=" * 70)

class SimpleNeuralNetwork:
    """Simple 2-layer neural network"""
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, learning_rate):
        # Output layer error
        delta2 = (y - self.a2) * self.sigmoid_derivative(self.a2)

        # Hidden layer error
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)

        # Update weights
        self.W2 += learning_rate * np.dot(self.a1.T, delta2)
        self.b2 += learning_rate * np.sum(delta2, axis=0, keepdims=True)
        self.W1 += learning_rate * np.dot(X.T, delta1)
        self.b1 += learning_rate * np.sum(delta1, axis=0, keepdims=True)

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for _ in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)

# Train on XOR
nn = SimpleNeuralNetwork(2, 4, 1)
nn.train(X_xor, y_xor.reshape(-1, 1), epochs=10000, learning_rate=0.5)

print("\nMLP Predictions on XOR:")
for x in X_xor:
    pred = nn.predict([x])[0][0]
    print(f"{x} -> {pred:.4f} (rounded: {round(pred)})")

# ============================================================================
# 6. GRADIENT DESCENT
# ============================================================================

print("\n" + "=" * 70)
print("6. GRADIENT DESCENT")
print("=" * 70)

print("""
Gradient Descent: Optimization algorithm

The Goal: Minimize loss function

Process:
1. Calculate gradient (direction of steepest increase)
2. Move in opposite direction (decrease loss)
3. Repeat until convergence

Learning Rate:
- Too small: slow convergence
- Too large: overshoot, may not converge

Variants:
- Batch GD: Use all data per iteration
- Stochastic GD (SGD): Use one sample per iteration
- Mini-batch GD: Use small batch per iteration
""")

# Visualize gradient descent
def f(x):
    return x**2

def gradient(x):
    return 2 * x

x = 10  # Start far from minimum
learning_rate = 0.1

trajectory = [x]
for _ in range(20):
    x = x - learning_rate * gradient(x)
    trajectory.append(x)

plt.figure(figsize=(10, 5))
x_range = np.linspace(-12, 12, 100)
plt.plot(x_range, f(x_range), 'b-', label='f(x) = x²')
plt.plot(trajectory, f(np.array(trajectory)), 'ro-', markersize=5, label='Gradient Descent')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.title('Gradient Descent Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 7. LOSS FUNCTIONS
# ============================================================================

print("\n" + "=" * 70)
print("7. LOSS FUNCTIONS")
print("=" * 70)

print("""
Loss Functions: Measure how well the model is doing

1. Mean Squared Error (MSE):
   MSE = (1/n) * Σ(y_pred - y_actual)²
   - For regression
   - Penalizes large errors heavily

2. Cross-Entropy (Log Loss):
   CE = -Σ y_actual * log(y_pred)
   - For classification
   - Good for probabilistic outputs

3. Binary Cross-Entropy:
   BCE = -[y * log(p) + (1-y) * log(1-p)]
   - For binary classification

4. Categorical Cross-Entropy:
   - For multi-class classification
""")

# ============================================================================
# 8. BACKPROPAGATION
# ============================================================================

print("\n" + "=" * 70)
print("8. BACKPROPAGATION")
print("=" * 70)

print("""
Backpropagation: Efficient way to compute gradients

Process:
1. Forward pass: Compute predictions
2. Compute loss
3. Backward pass: Compute gradients layer by layer
4. Update weights

Chain Rule:
- ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w
- Gradients flow from output to input
- Each layer uses the gradient from the next layer

This is why neural networks can learn complex patterns!
""")

# ============================================================================
# 9. PRACTICAL EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL EXAMPLE: CLASSIFICATION")
print("=" * 70)

from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Train neural network
nn = SimpleNeuralNetwork(2, 8, 1)
nn.train(X, y.reshape(-1, 1), epochs=5000, learning_rate=0.1)

# Visualize decision boundary
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                      np.linspace(y_min, y_max, 100))

Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0.5], linestyles=['--'])
plt.title('Neural Network Decision Boundary')
plt.show()

# Accuracy
predictions = (nn.predict(X) > 0.5).astype(int).flatten()
accuracy = np.mean(predictions == y)
print(f"Training Accuracy: {accuracy:.4f}")

print("\n" + "=" * 70)
print("NEURAL NETWORK FUNDAMENTALS SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. Perceptron is the simplest neural network
2. Activation functions add non-linearity
3. Hidden layers enable learning complex patterns
4. Gradient descent optimizes the network
5. Backpropagation computes gradients efficiently
6. MLP can solve non-linear problems (like XOR)

Next Steps:
- Learn to use Keras/TensorFlow for easier implementation
- Explore different architectures (CNN, RNN)
- Understand regularization techniques
""")
