from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize từ scratch
mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)

X_train_scaled = (X_train - mu) / sigma
X_test_scaled = (X_test - mu) / sigma

# Train SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')
# model.fit(X_train_scaled, y_train)
#
# # Predict
# y_pred = model.predict(X_test_scaled)
#
# # Kết quả
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print()
# print(classification_report(y_test, y_pred, target_names=data.target_names))

# Thử các giá trị C khác nhau
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model = SVC(kernel='rbf', C=C, gamma='scale')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("C =", C, "→ Accuracy:", round(acc * 100, 2), "%")

# Thử các kernel khác nhau
for kernel in ['linear', 'rbf', 'poly']:
    model = SVC(kernel=kernel, C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("kernel =", kernel, "→ Accuracy:", round(acc * 100, 2), "%")