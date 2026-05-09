import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

np.random.seed(42)
n_input, n_hidden = 2, 4

# Proper weight matrices instead of scalars
w1 = np.random.randn(n_input, n_hidden)  # (2, 4)
b1 = np.zeros(n_hidden)                          # (4,)
w2 = np.random.randn(n_hidden)          # (4,)
b2 = 0.0

learning_rate = 0.1

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # (4, 2)
y_true = np.array([0, 1, 1, 0])                  # (4,)

m = x.shape[0]  # number of samples

print(f"{'Epoch':<10} | {'Loss':<10} | {'Predictions'}")
print("-" * 55)

for epoch in range(1, 10000):
    # Forward pass — all samples at once
    z1 = np.dot(x, w1) + b1   # (4, 4)
    a1 = relu(z1)               # (4, 4)
    z2 = np.dot(a1, w2) + b2   # (4,)
    p  = sigmoid(z2)            # (4,)

    loss = binary_cross_entropy(y_true, p)

    # Backward pass — gradients summed over all samples
    delta2 = (p - y_true) * (p * (1 - p))        # (4,)
    dw2 = np.dot(a1.T, delta2) / m               # (4,)  — averaged
    db2 = np.mean(delta2)

    delta1 = np.outer(delta2, w2) * relu_derivative(z1)  # (4, 4)
    dw1 = np.dot(x.T, delta1) / m               # (2, 4) — averaged
    db1 = np.mean(delta1, axis=0)               # (4,)

    # Single weight update per epoch using averaged gradients
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    if epoch == 1:
        print(f"z1 shape: {z1.shape}  | values:\n{z1}\n")
        print(f"a1 shape: {a1.shape}  | values:\n{a1}\n")
        print(f"z2 shape: {z2.shape}  | values: {z2}\n")
        print(f"p  shape: {p.shape}   | values: {p}\n")
        print("-" * 55)

    if epoch % 1000 == 0:
        print(f"{epoch:<10} | {loss:<10.4f} | {np.round(p, 3)}")