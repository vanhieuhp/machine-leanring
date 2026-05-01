import math
import matplotlib.pyplot as plt

x, y = 2.0, 1
a, b, lr = 0.0, 0.0, 0.1
sigmoid = lambda z: 1 / (1 + math.exp(-z))

losses = []
for _ in range(100):
    z = a * x + b
    P = sigmoid(z)
    losses.append(-(y * math.log(P) + (1 - y) * math.log(1 - P)))
    a -= lr * (P - y) * x
    b -= lr * (P - y)

plt.plot(losses, color='royalblue', linewidth=2)
plt.xlabel("Vòng lặp")
plt.ylabel("Loss")
plt.title("Loss giảm dần theo Gradient Descent")
plt.grid(alpha=0.3)
plt.show()
