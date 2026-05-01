from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print(f"Train accuracy: {clf.score(X_train, y_train):.2%}")
print(f"Test accuracy:  {clf.score(X_test, y_test):.2%}")

for depth in [1, 2, 3, 4, 5, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc  = clf.score(X_test, y_test)
    print(f"depth={str(depth):4s} → train={train_acc:.2%}, test={test_acc:.2%}")