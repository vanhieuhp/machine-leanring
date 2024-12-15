import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

# m is the number of training examples
print(f"m: {x_train.shape[0]}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

w = 200
b = 100
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)
plt.plot(x_train, tmp_f_wb, label="Our prediction", color='blue')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.title('Price of house')
plt.xlabel('Size of house')
plt.ylabel('Price of house')
plt.legend()
plt.show()

x_i = 1.2
cost_1200sqrt = w * x_i + b
print(f"Cost of house with size 1200 sqrt is: {cost_1200sqrt}")