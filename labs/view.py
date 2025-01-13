import numpy as np
import matplotlib.pyplot as plt

# Define the sequence a_n = 1/n
n_values = np.arange(1, 21)  # From n = 1 to n = 20
a_n_values = 1 / n_values

# Plotting the sequence
plt.figure(figsize=(8, 6))
plt.plot(n_values, a_n_values, marker='o', linestyle='-', color='b', label="a_n = 1/n")
plt.axhline(0, color='r', linestyle='--', label="Chặn dưới (L = 0)")
plt.title("Dãy số giảm a_n = 1/n hội tụ về 0")
plt.xlabel("n (chỉ số dãy số)")
plt.ylabel("a_n (giá trị dãy số)")
plt.legend()
plt.grid(True)
plt.show()
