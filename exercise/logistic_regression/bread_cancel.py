from sklearn.datasets import load_breast_cancer
import math
import random
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

print("Số bệnh nhân  :", data.data.shape[0])
print("Số features   :", data.data.shape[1])
print("Tên features  :", data.feature_names[:5], "...")
print("Nhãn          :", data.target_names)
print("Phân bố nhãn  :", "lành tính =", sum(data.target == 1), "| ác tính =", sum(data.target == 0))

X_raw   = data.data.tolist()
y_raw   = data.target.tolist()

# ── 2. Chuẩn hóa features (StandardScaler từ tay) ────────────────
# Lý do: 30 features có đơn vị rất khác nhau
# mean radius ~ 14, mean area ~ 654 — nếu không chuẩn hóa
# gradient của feature lớn sẽ áp đảo feature nhỏ
n_samples  = len(X_raw)
n_features = len(X_raw[0])

# Tính mean của từng feature
mean = []
for j in range(n_features):
    total = 0.0
    for i in range(n_samples):
        total += X_raw[i][j]
    mean.append(total / n_samples)

# Tính std của từng feature
std = []
for j in range(n_features):
    total_sq = 0.0
    for i in range(n_samples):
        diff = X_raw[i][j] - mean[j]
        total_sq = total_sq + diff * diff
    std.append(math.sqrt(total_sq / n_samples))

# Chuẩn hóa: x_new = (x - mean) / std
X_scaled = []
for i in range(n_samples):
    row = []
    for j in range(n_features):
        x_new = (X_raw[i][j] - mean[j]) / std[j]
        row.append(x_new)
    X_scaled.append(row)

# ── 3. Train / Test split (80/20) ────────────────────────────────
random.seed(42)
indices  = list(range(n_samples))
random.shuffle(indices)

split    = int(0.8 * n_samples)
train_idx = indices[:split]
test_idx  = indices[split:]

X_train = []
y_train = []
for i in train_idx:
    X_train.append(X_scaled[i])
    y_train.append(y_raw[i])

X_test = []
y_test = []
for i in test_idx:
    X_test.append(X_scaled[i])
    y_test.append(y_raw[i])

print(f"Train: {len(X_train)} bệnh nhân")
print(f"Test : {len(X_test)} bệnh nhân")

from sklearn.linear_model import LogisticRegression as SklearnLR
sk_model = SklearnLR(max_iter=1000)
sk_model.fit(X_train, y_train)

print(f"  Sklearn : {sk_model.score(X_test, y_test):.2%}")