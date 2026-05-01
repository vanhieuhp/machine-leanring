from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier

# Dataset phức tạp hơn — 1000 mẫu, 20 features, nhiều noise
X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=10, n_redundant=5,
                            random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Decision Tree:")
print(f"  Train: {dt.score(X_train, y_train):.2%}")
print(f"  Test:  {dt.score(X_test, y_test):.2%}")

print("Random Forest:")
print(f"  Train: {rf.score(X_train, y_train):.2%}")
print(f"  Test:  {rf.score(X_test, y_test):.2%}")
