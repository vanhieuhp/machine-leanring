import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

# Load Heart Disease dataset
heart = fetch_openml(name='heart-c', version=1, as_frame=True)
df = heart.frame

print(df.head())
print(df.shape)
# 2. Kiểm tra tên cột mục tiêu thực tế
# target_column = heart.target_names[0]
# print(f"Tên cột mục tiêu thực tế là: {target_column}")
#
# # 3. Sử dụng tên cột đó để tính toán
# print(df[target_column].value_counts())

import matplotlib.pyplot as plt

# Xem thông tin tổng quan
print(df.info())
print("\n")
print(df.isnull().sum())

# Drop các rows có missing values
# Convert target num thành binary: <50 = 0, >50_1 = 1
# Print df.shape sau khi xử lý để confirm
df = df.dropna()

df["num"] = (df["num"] == ">50_1").astype(int)
print(df.shape)

X = df.drop("num", axis=1)
y = df["num"]

X = pd.get_dummies(X)
print(X.shape)
print(X.columns.tolist())