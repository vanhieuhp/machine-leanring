import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Dataset: 6 bệnh nhân
# x = gene expression, y = có bệnh (1) hay không (0)
data = [
    (0.5, 0),
    (1.0, 0),
    (1.5, 0),
    (2.5, 1),
    (3.0, 1),
    (3.5, 1),
]

# Khởi tạo
a, b = 0.0, 0.0
lr = 0.1
n = len(data)

print(f"{'Vòng':>5} | {'a':>7} | {'b':>7} | {'Loss':>7}")
print("-" * 35)

for epoch in range(1, 101):
    predictions = []
    for x, y in data:
        z = a * x + b
        P = sigmoid(z)
        predictions.append((x, y, P))

    # Tính loss trung bình
    total_loss = 0
    for x, y, P in predictions:
        total_loss += -(y * math.log(P) + (1 - y) * math.log(1 - P))
    avg_loss = total_loss / n

    # Backward pass — gradient trung bình
    grad_a_list = []
    grad_b_list = []

    for x, y, P in predictions:
        grad_a = (P - y) * x
        grad_b = (P - y)
        grad_a_list.append(grad_a)
        grad_b_list.append(grad_b)

    grad_a_new = sum(grad_a_list) / n
    grad_b_new = sum(grad_b_list) / n

    a = a - lr * grad_a_new
    b = b - lr * grad_b_new
    if epoch <= 5 or epoch % 20 == 0:
        print(f"{epoch:>5} | {a:>7.4f} | {b:>7.4f} | {avg_loss:>7.4f}")

print("\nDự đoán sau training:")
print(f"{'x':>5} | {'y thật':>8} | {'P dự đoán':>10} | {'Kết quả':>10}")
print("-" * 42)
for x, y in data:
    z = a * x + b
    P = sigmoid(z)
    label = "đúng ✓" if (P > 0.5) == bool(y) else "sai ✗"
    print(f"{x:>5} | {y:>8} | {P:>10.4f} | {label:>10}")
