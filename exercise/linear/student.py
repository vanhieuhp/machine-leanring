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

model.fit(X_train, y_train)