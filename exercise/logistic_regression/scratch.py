import math

# ── Dữ liệu: 1 bệnh nhân ──────────────────────────
x = 2.0   # gene expression
y = 1     # có bệnh

# ── Khởi tạo tham số ──────────────────────────────
a = 0.0
b = 0.0
lr = 0.1  # learning rate

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# ── Vòng lặp Gradient Descent ─────────────────────
print(f"{'Vòng':>5} | {'a':>7} | {'b':>7} | {'P':>7} | {'Loss':>8}")
print("-" * 45)

for i in range(20):
    # Forward pass
    z    = a * x + b
    P    = sigmoid(z)
    loss = -(y * math.log(P) + (1 - y) * math.log(1 - P))

    # Backward pass
    dL_da = (P - y) * x
    dL_db = (P - y)

    # Cập nhật
    a = a - lr * dL_da
    b = b - lr * dL_db

    if i < 10 or i == 19:
        print(f"{i+1:>5} | {a:>7.4f} | {b:>7.4f} | {P:>7.4f} | {loss:>8.4f}")