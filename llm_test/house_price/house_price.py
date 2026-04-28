import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/train.csv')
df_numeric = df.select_dtypes(include=[np.number]).dropna()

features = ['OverallQual', 'GrLivArea', 'GarageCars']
X = df_numeric[features]
y = df_numeric['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Vẽ 1 feature vs SalePrice + đường dự đoán
# x_plot = df_numeric['GrLivArea']
# y_plot = df_numeric['SalePrice']
#
# model_simple = LinearRegression()
# model_simple.fit(df_numeric[['GrLivArea']], y_plot)
#
# x_line = np.linspace(x_plot.min(), x_plot.max(), 100).reshape(-1, 1)
# y_line = model_simple.predict(x_line)
#
# plt.figure(figsize=(10, 6))
# plt.scatter(x_plot, y_plot, alpha=0.3, color='steelblue', label='Dữ liệu thực')
# plt.plot(x_line, y_line, color='red', linewidth=2, label='Đường dự đoán')
# plt.xlabel('Diện tích sống (sqft)')
# plt.ylabel('Giá nhà ($)')
# plt.title('Linear Regression — 1 feature')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Vẽ Loss surface đơn giản
# Giả sử chỉ có 1 hệ số cần tìm: slope (hệ số của GrLivArea)

x_data = df_numeric['GrLivArea'].values
y_data = df_numeric['SalePrice'].values

slopes = np.linspace(0, 200, 300)
print(slopes)
losses = []

for slope in slopes:
    y_pred_temp = slope * x_data
    mse = np.mean((y_data - y_pred_temp) ** 2)
    losses.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(slopes, losses, color='steelblue', linewidth=2)
plt.xlabel('Hệ số slope')
plt.ylabel('MSE Loss')
plt.title('Loss thay đổi theo hệ số — hình dạng như ngọn đồi lộn ngược')
plt.tight_layout()
plt.show()

# Mô phỏng Gradient Descent thủ công
x_data = df_numeric['GrLivArea'].values
y_data = df_numeric['SalePrice'].values

# Khởi tạo
slope = 0.0
learning_rate = 0.000001
n = len(x_data)

history = []

for step in range(100):
    # Dự đoán
    y_pred_temp = slope * x_data

    # Tính gradient (đạo hàm của MSE theo slope)
    gradient = (-2 / n) * np.sum(x_data * (y_data - y_pred_temp))

    # Cập nhật slope
    slope = slope - learning_rate * gradient

    # Lưu lại
    mse = np.mean((y_data - y_pred_temp) ** 2)
    history.append((step, slope, mse))

# Vẽ quá trình học
steps = [h[0] for h in history]
slopes_history = [h[1] for h in history]
losses_history = [h[2] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(steps, slopes_history, color='steelblue', linewidth=2)
ax1.set_xlabel('Bước')
ax1.set_ylabel('Slope')
ax1.set_title('Slope thay đổi theo từng bước')

ax2.plot(steps, losses_history, color='red', linewidth=2)
ax2.set_xlabel('Bước')
ax2.set_ylabel('MSE Loss')
ax2.set_title('Loss giảm dần theo từng bước')

plt.tight_layout()
plt.show()

print(f"Slope cuối cùng: {slope:.2f}")