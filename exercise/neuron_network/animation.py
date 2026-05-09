"""
Neural Network từ Scratch — Phase 2
=====================================
Eric Nguyen Van — ML Curriculum
Buổi 1-2: Perceptron → Multi-layer NN → XOR Problem

Nội dung:
- Activation functions: ReLU, Sigmoid
- Forward pass
- Loss function: Binary Cross Entropy
- Backward pass (Backpropagation)
- Training loop
- Visualize decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. ACTIVATION FUNCTIONS
# ============================================================

def relu(z):
    """
    ReLU: max(0, z)
    - Nếu z >= 0 → giữ nguyên
    - Nếu z < 0  → trả về 0
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Đạo hàm của ReLU:
    - Nếu z > 0 → 1
    - Nếu z <= 0 → 0
    Dùng trong backpropagation
    """
    return np.where(z > 0, 1.0, 0.0)


def sigmoid(z):
    """
    Sigmoid: 1 / (1 + e^(-z))
    - Output nằm trong (0, 1)
    - Dùng cho output layer (binary classification)
    """
    return 1 / (1 + np.exp(-z))


# ============================================================
# 2. FORWARD PASS
# ============================================================

def forward_hidden(x, w, b):
    """
    Forward pass qua hidden layer — dùng ReLU
    x: input
    w: weight matrix
    b: bias
    """
    z = np.dot(x, w) + b
    a = relu(z)
    return z, a


def forward_output(a, w, b):
    """
    Forward pass qua output layer — dùng Sigmoid
    a: activation từ layer trước
    w: weight
    b: bias
    """
    z = np.dot(a, w) + b
    p = sigmoid(z)
    return z, p


# ============================================================
# 3. LOSS FUNCTION
# ============================================================

def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross Entropy Loss:
    Loss = -[y * log(p) + (1-y) * log(1-p)]

    Clip y_pred để tránh log(0)
    """
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


# ============================================================
# 4. BACKWARD PASS (BACKPROPAGATION)
# ============================================================

def calculate_gradients(x, z1, a1, p, y_true, w2):
    """
    Tính gradient bằng chain rule — lan truyền ngược từ output về input

    Output layer:
        delta2 = (p - y) * sigmoid'(z2) = (p - y) * p(1-p)
        dW2 = a1.T @ delta2 / m
        db2 = mean(delta2)

    Hidden layer:
        delta1 = (delta2 @ w2.T) * relu'(z1)
        dW1 = x.T @ delta1 / m
        db1 = mean(delta1)
    """
    m = x.shape[0]  # số sample

    # --- Output layer ---
    error = p - y_true
    sigmoid_derivative = p * (1 - p)
    delta2 = error * sigmoid_derivative

    dw2 = np.dot(a1.T, delta2) / m
    db2 = np.mean(delta2)

    # --- Hidden layer ---
    relu_deriv = relu_derivative(z1)
    delta1 = np.outer(delta2, w2) * relu_deriv

    dw1 = np.dot(x.T, delta1) / m
    db1 = np.mean(delta1, axis=0)

    return dw1, db1, dw2, db2


# ============================================================
# 5. UPDATE WEIGHTS
# ============================================================

def update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    """
    Gradient Descent:
    w = w - learning_rate * gradient
    """
    w1_new = w1 - learning_rate * dw1
    b1_new = b1 - learning_rate * db1
    w2_new = w2 - learning_rate * dw2
    b2_new = b2 - learning_rate * db2
    return w1_new, b1_new, w2_new, b2_new


# ============================================================
# 6. TRAINING — XOR PROBLEM
# ============================================================

# Dataset XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

# Hyperparameters
n_input  = 2
n_hidden = 4
n_output = 1
learning_rate = 0.5
epochs = 10000

# Khởi tạo weight ngẫu nhiên (KHÔNG nhân 0.1 — tránh vanishing gradient)
np.random.seed(42)
w1 = np.random.randn(n_input, n_hidden)   # shape (2, 4)
b1 = np.zeros(n_hidden)                    # shape (4,)
w2 = np.random.randn(n_hidden)             # shape (4,)
b2 = 0.0

# Lưu loss để plot
loss_history = []

print(f"{'Epoch':<10} | {'Loss':<10} | {'Predictions'}")
print("-" * 60)

for epoch in range(1, epochs + 1):

    # --- Forward pass ---
    z1, a1 = forward_hidden(X, w1, b1)
    z2, p  = forward_output(a1, w2, b2)

    # --- Loss ---
    loss = binary_cross_entropy(y, p)
    loss_history.append(loss)

    # --- Backward pass ---
    dw1, db1, dw2, db2 = calculate_gradients(X, z1, a1, p, y, w2)

    # --- Update weights ---
    w1, b1, w2, b2 = update_weights(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

    # --- Log ---
    if epoch % 1000 == 0:
        print(f"{epoch:<10} | {loss:<10.4f} | {np.round(p, 3)}")

print("\nKết quả cuối:")
print(f"  [0,0] → {p[0]:.4f} (đúng: 0)")
print(f"  [0,1] → {p[1]:.4f} (đúng: 1)")
print(f"  [1,0] → {p[2]:.4f} (đúng: 1)")
print(f"  [1,1] → {p[3]:.4f} (đúng: 0)")


# ============================================================
# 7. VISUALIZE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Loss curve ---
axes[0].plot(loss_history, color='steelblue', linewidth=1.5)
axes[0].set_title('Loss qua các Epoch', fontsize=13)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross Entropy Loss')
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Decision Boundary ---
x1_range = np.linspace(-0.5, 1.5, 300)
x2_range = np.linspace(-0.5, 1.5, 300)
xx1, xx2 = np.meshgrid(x1_range, x2_range)

grid = np.column_stack([xx1.ravel(), xx2.ravel()])

_, a1_grid = forward_hidden(grid, w1, b1)
_, p_grid  = forward_output(a1_grid, w2, b2)
p_grid = p_grid.reshape(xx1.shape)

axes[1].contourf(xx1, xx2, p_grid, levels=50, cmap='RdBu', alpha=0.8)
axes[1].contour(xx1, xx2, p_grid, levels=[0.5], colors='white', linewidths=2.5)

colors = ['red' if label == 0 else 'blue' for label in y]
axes[1].scatter(X[:, 0], X[:, 1], c=colors, s=250, zorder=5,
                edgecolors='white', linewidths=2)

labels = ['(0,0)\ny=0', '(0,1)\ny=1', '(1,0)\ny=1', '(1,1)\ny=0']
offsets = [(0.05, 0.07), (0.05, 0.07), (0.05, -0.12), (0.05, 0.07)]
for i, (xi, yi_coord) in enumerate(X):
    axes[1].annotate(labels[i], (xi, yi_coord),
                     xytext=(xi + offsets[i][0], yi_coord + offsets[i][1]),
                     fontsize=10, color='white', fontweight='bold')

axes[1].set_title('Decision Boundary — Neural Network giải XOR', fontsize=13)
axes[1].set_xlabel('x₁')
axes[1].set_ylabel('x₂')

plt.tight_layout()
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "neural_network_xor.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nĐã lưu hình: {output_path}")
