import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

from svm.scratch import y_pred

# 1. Load data
heart = fetch_openml(name='heart-c', version=1, as_frame=True)
df = heart.frame

# 2. Kiểm tra
print(df.shape)
print(df.isnull().sum())

# 3. Drop missing rows
df = df.dropna()

# 4. Convert target thành binary
df["num"] = (df["num"] == ">50_1").astype(int)

# 5. Tách features và target
X = df.drop(columns=["num"])
y = df["num"]

# 6. One-hot encode categorical columns
X = pd.get_dummies(X)

print(X.shape)
print(X.columns.tolist())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train và evaluate từng model
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"{name}: {acc * 100:.2f}% - Precision: {precision * 100:.2f}% - Recall: {recall * 100:.2f}%")
