import math

import numpy as np

A = [45, 48, 50, 52, 55]
B = [10, 20, 50, 80, 90]

mean_a = sum(A) / len(A)
mean_b = sum(B) / len(B)

print(mean_a, mean_b)

sigma_a = 0
sigma_b = 0

sum_a = 0
for i, a in enumerate(A):
    sum_a += (a-mean_a)**2

sigma_a = math.sqrt(sum_a / len(A))

sum_b = 0
for i, b in enumerate(B):
    sum_b += (b-mean_b)**2

sigma_b = math.sqrt(sum_b / len(B))

print(sigma_a, sigma_b)

mean_a = np.mean(A)
mean_b = np.mean(B)

sigma_a = np.std(A)
sigma_b = np.std(B)

print(f"mean_a: {mean_a}, mean_b: {mean_b}")
print(f"sigma_a: {sigma_a}, sigma_b: {sigma_b}")


