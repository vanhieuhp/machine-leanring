from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import numpy as np


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

# Định nghĩa các giá trị cần thử
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 0.001, 0.01, 0.1]
}

# Grid Search với 5-Fold Cross Validation
grid_search = GridSearchCV(
    SVC(kernel='rbf'),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Kết quả
print("Best params:", grid_search.best_params_)
print("Best CV score:", round(grid_search.best_score_ * 100, 2), "%")

# Đánh giá trên test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print("Test accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")