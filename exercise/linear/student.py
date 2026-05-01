import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

model = LinearRegression()

# Tạo dữ liệu giả
np.random.seed(42)
n = 100  # 100 học sinh

gio_hoc = np.random.uniform(1, 10, n)
diem_giua_ki = np.random.uniform(3, 10, n)

# Điểm cuối kì = quy luật thực + nhiễu nhỏ
diem_cuoi_ki = 0.5 * gio_hoc + 0.4 * diem_giua_ki + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    'gio_hoc': gio_hoc,
    'diem_giua_ki': diem_giua_ki,
    'diem_cuoi_ki': diem_cuoi_ki
})

print(df.head())
print(f"Shape: {df.shape}")

# Bước 1: Tách X và y trước
X = df[['gio_hoc', 'diem_giua_ki']]
y = df['diem_cuoi_ki']

# Bước 2: Chia train vs (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Bước 3: Chia (val + test) thành val và test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 1. Train model với X_train, y_train
slope = 0.0
learning_rate = 0.000001
n = len(X_train)
for step in range(1000):
    y_pred = slope * X_train['gio_hoc'] + slope * X_train['diem_giua_ki']
    error = y_pred - y_train
    gradient = (2/n) * np.sum(error)
    slope -= learning_rate * gradient

print(f"Hệ số slope sau 1000 bước: {slope}")
# 2. Dự đoán trên X_train → y_pred_train
y_pred_train = slope * X_train['gio_hoc'] + slope * X_train['diem_giua_ki']
print(f"y_pred_train: {y_pred_train}")
print(f"y_train: {y_train}")

# 3. Dự đoán trên X_val → y_pred_val
y_pred_val = slope * X_val['gio_hoc'] + slope * X_val['diem_giua_ki']
print(f"y_pred_val: {y_pred_val}")
print(f"y_val: {y_val}")

# 4. Dự đoán trên X_test → y_pred_test
y_pred_test = slope * X_test['gio_hoc'] + slope * X_test['diem_giua_ki']
print(f"y_pred_test: {y_pred_test}")
print(f"y_test: {y_test}")
# 5. Tính MSE trên y_pred_train, y_train → train_mse
train_mse = mean_squared_error(y_train, y_pred_train)
print(f"Train MSE: {train_mse}")

# 6. Tính MSE trên y_pred_val, y_val → val_mse
val_mse = mean_squared_error(y_val, y_pred_val)
print(f"Validation MSE: {val_mse}")

# 7. Tính MSE trên y_pred_test, y_test → test_mse
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE: {test_mse}")
