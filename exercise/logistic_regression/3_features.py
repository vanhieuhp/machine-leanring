import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def dot(a_list, x_list):
    tuples = zip(a_list, x_list)
    sums = 0
    for a, x in tuples:
        sums += a * x
    return sums

# Dataset: 6 bệnh nhân, 3 features
# [tuổi chuẩn hóa, gene_A, gene_B], nhãn y
data = [
    ([0.2, 0.1, 0.4], 0),
    ([0.3, 0.2, 0.5], 0),
    ([0.4, 0.3, 0.3], 0),
    ([0.6, 0.7, 0.6], 1),
    ([0.7, 0.8, 0.8], 1),
    ([0.9, 0.9, 0.7], 1),
]

# Khởi tạo — 3 hệ số + 1 intercept
a = [0.0, 0.0, 0.0]
b = 0.0
lr = 0.1
n = len(data)
print(f"{'Vòng':>5} | {'a1':>7} | {'a2':>7} | {'a3':>7} | {'b':>7} | {'Loss':>7}")
print("-" * 55)

for epoch in range(1, 1000):
    preds = []
    for x_list, y in data:
        z = dot(a, x_list) + b
        P = sigmoid(z)
        preds.append((x_list, y, P))

    # avg_loss
    total_loss = 0
    for x_list, y, P in preds:
        total_loss += -(y * math.log(P) + (1 - y) * math.log(1 - P))
    avg_loss = total_loss / n

    # backward pass
    grad_a = [0.0] * len(a)
    grad_b = 0.0

    for x_list, y, P in preds:
        err = P - y
        for i in range(len(a)):
            grad_a[i] += err * x_list[i]
        grad_b += err

    grad_a = [g/n for g in grad_a]
    grad_b /= n


    for i in range(len(a)):
        a[i] = a[i] - lr * grad_a[i]

    b = b - lr * grad_b
    if epoch <= 5 or epoch % 20 == 0:
        print(f"{epoch:>5} | {a[0]:>7.4f} | {a[1]:>7.4f} | {a[2]:>7.4f} | {b:>7.4f} | {avg_loss:>7.4f}")

print("\nDự đoán sau training:")
# print(f"{'x':>5} | {'y thật':>8} | {'P dự đoán':>10} | {'Kết quả':>10}")
print(f"{'x1':>5} | {'x2':>5} | {'x3':>5} | {'y':>5} | {'P':>10} | {'Kết quả':>10}")

print("-" * 42)
for x_list, y in data:
    z = dot(a, x_list) + b
    P = sigmoid(z)
    label = "đúng ✓" if (P > 0.5) == bool(y) else "sai ✗"
    print(f"{x_list[0]:>5.2f} | {x_list[1]:>5.2f} | {x_list[2]:>5.2f} | {y:>5} | {P:>10.4f} | {label:>10}")

