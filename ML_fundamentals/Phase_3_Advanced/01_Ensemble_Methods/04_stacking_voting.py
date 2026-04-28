"""
Stacking & Voting Ensembles
============================

This module covers:
- Voting Classifier (Hard and Soft)
- Stacking Classifier
- Building custom ensembles
- Combining diverse models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# 1. VOTING CLASSIFIER - CONCEPT
# ============================================================================

print("=" * 70)
print("1. VOTING CLASSIFIER CONCEPT")
print("=" * 70)

print("""
Voting Classifier:
- Train multiple models
- Combine predictions via voting
- Two types:
  1. Hard Voting: Majority vote (class with most votes wins)
  2. Soft Voting: Average probabilities (class with highest probability wins)

When to use:
- When you have diverse models
- When you want simplicity
- As a baseline ensemble
""")

# ============================================================================
# 2. HARD VOTING
# ============================================================================

print("\n" + "=" * 70)
print("2. HARD VOTING")
print("=" * 70)

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=3, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
clf2 = DecisionTreeClassifier(max_depth=5, random_state=42)
clf3 = KNeighborsClassifier(n_neighbors=5)

# Hard Voting
hard_voting = VotingClassifier(
    estimators=[
        ('rf', clf1),
        ('dt', clf2),
        ('knn', clf3)
    ],
    voting='hard'
)

hard_voting.fit(X_train, y_train)
y_pred_hard = hard_voting.predict(X_test)

print(f"Hard Voting Accuracy: {accuracy_score(y_test, y_pred_hard):.4f}")

# Individual model accuracies
for name, clf in [('Random Forest', clf1), ('Decision Tree', clf2), ('KNN', clf3)]:
    clf.fit(X_train, y_train)
    print(f"{name}: {accuracy_score(y_test, clf.predict(X_test)):.4f}")

# ============================================================================
# 3. SOFT VOTING
# ============================================================================

print("\n" + "=" * 70)
print("3. SOFT VOTING")
print("=" * 70)

print("""
Soft Voting:
- Uses predicted probabilities
- Average probabilities across models
- Class with highest average probability wins
- Requires all models to support predict_proba
""")

# Need classifiers that support predict_proba
clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
clf2 = LogisticRegression(max_iter=1000, random_state=42)
clf3 = GaussianNB()

soft_voting = VotingClassifier(
    estimators=[
        ('rf', clf1),
        ('lr', clf2),
        ('nb', clf3)
    ],
    voting='soft'
)

soft_voting.fit(X_train, y_train)
y_pred_soft = soft_voting.predict(X_test)

print(f"Soft Voting Accuracy: {accuracy_score(y_test, y_pred_soft):.4f}")

# Individual model accuracies
print("\nIndividual Model Accuracies:")
for name, clf in [('Random Forest', clf1), ('Logistic Regression', clf2), ('Naive Bayes', clf3)]:
    clf.fit(X_train, y_train)
    print(f"  {name}: {accuracy_score(y_test, clf.predict(X_test)):.4f}")

# ============================================================================
# 4. VOTING WITH DIFFERENT WEIGHTS
# ============================================================================

print("\n" + "=" * 70)
print("4. WEIGHTED VOTING")
print("=" * 70)

print("""
Weighted Voting:
- Give more weight to better models
- Weights should sum to 1
- Higher weight = more influence
""")

clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf3 = LogisticRegression(max_iter=1000, random_state=42)

# Weighted voting
weighted_voting = VotingClassifier(
    estimators=[
        ('rf', clf1),
        ('gb', clf2),
        ('lr', clf3)
    ],
    voting='soft',
    weights=[0.5, 0.3, 0.2]  # RF gets more weight
)

weighted_voting.fit(X_train, y_train)
y_pred_weighted = weighted_voting.predict(X_test)

print(f"Weighted Voting Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")

# ============================================================================
# 5. STACKING - CONCEPT
# ============================================================================

print("\n" + "=" * 70)
print("5. STACKING CONCEPT")
print("=" * 70)

print("""
Stacking (Stacked Generalization):
- Two-level approach:
  Level 0 (Base Models): Train multiple diverse models
  Level 1 (Meta-Learner): Train a meta-model on base model predictions

Why it works:
- Base models make different errors
- Meta-learner learns to combine them optimally
- Can capture patterns individual models miss

Workflow:
1. Split data into K folds
2. For each base model:
   - Train on K-1 folds
   - Predict on held-out fold
   - Repeat to get out-of-fold predictions
3. Stack all base model predictions
4. Train meta-learner on stacked predictions
""")

# ============================================================================
# 6. BASIC STACKING
# ============================================================================

print("\n" + "=" * 70)
print("6. BASIC STACKING")
print("=" * 70)

# Define base classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('nb', GaussianNB())
]

# Stacking with Logistic Regression as meta-learner
stacking_clf = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,  # 5-fold cross-validation
    passthrough=False  # Don't pass original features to meta-learner
)

stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")

# ============================================================================
# 7. STACKING WITH MORE FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("7. STACKING WITH ORIGINAL FEATURES")
print("=" * 70)

print("""
passthrough=True:
- Meta-learner gets both:
  1. Base model predictions
  2. Original input features
- Can capture interactions
- More information for meta-learner
""")

stacking_with_features = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=True  # Include original features
)

stacking_with_features.fit(X_train, y_train)
y_pred_stack_features = stacking_with_features.predict(X_test)

print(f"Stacking with Features Accuracy: {accuracy_score(y_test, y_pred_stack_features):.4f}")

# ============================================================================
# 8. CHOOSING META-LEARNER
# ============================================================================

print("\n" + "=" * 70)
print("8. CHOOSING META-LEARNER")
print("=" * 70)

print("""
Common meta-learners:
1. Logistic Regression: Simple, regularized
2. Random Forest: Can capture non-linear combinations
3. XGBoost/LightGBM: Powerful meta-learner

Rules:
- Start with simple (Logistic Regression)
- Use regularized models to prevent overfitting
- Meta-learner should be different from base models
""")

# Different meta-learners
meta_learners = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=50, random_state=42))
]

print("\nComparing Meta-Learners:")
print("-" * 40)

for name, meta in meta_learners:
    stack = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta,
        cv=5
    )
    stack.fit(X_train, y_train)
    acc = accuracy_score(y_test, stack.predict(X_test))
    print(f"{name}: {acc:.4f}")

# ============================================================================
# 9. PRACTICAL ENSEMBLE EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL ENSEMBLE EXAMPLE")
print("=" * 70)

# Diverse classifiers for best results
diverse_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('nb', GaussianNB())
]

# Voting ensemble
voting_ensemble = VotingClassifier(
    estimators=diverse_classifiers,
    voting='soft'
)

# Stacking ensemble
stacking_ensemble = StackingClassifier(
    estimators=diverse_classifiers,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

print("\nComparing Ensembles:")
print("-" * 40)

# Test each ensemble
ensembles = [
    ('Voting (Soft)', voting_ensemble),
    ('Stacking', stacking_ensemble)
]

for name, ensemble in ensembles:
    ensemble.fit(X_train, y_train)
    acc = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"{name}: {acc:.4f}")

# ============================================================================
# 10. BAGGING AND BOOSTING
# ============================================================================

print("\n" + "=" * 70)
print("10. BAGGING AND BOOSTING")
print("=" * 70)

print("""
Bagging (Bootstrap Aggregating):
- Train models on bootstrap samples
- Average predictions
- Reduces variance
- Example: Random Forest

Boosting:
- Train models sequentially
- Each model focuses on errors
- Reduces bias
- Example: AdaBoost, Gradient Boosting
""")

# AdaBoost example
ada_clf = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

# Bagging with SVM
bagging_clf = BaggingClassifier(
    estimator=SVC(),
    n_estimators=10,
    random_state=42
)

print("\nBagging vs Boosting:")
print("-" * 40)

ada_clf.fit(X_train, y_train)
bagging_clf.fit(X_train, y_train)

print(f"AdaBoost: {accuracy_score(y_test, ada_clf.predict(X_test)):.4f}")
print(f"Bagging (SVM): {accuracy_score(y_test, bagging_clf.predict(X_test)):.4f}")

# ============================================================================
# 11. WHEN TO USE WHICH
# ============================================================================

print("\n" + "=" * 70)
print("11. WHEN TO USE WHICH")
print("=" * 70)

print("""
Quick Decision Guide:

1. Voting (Hard/Soft):
   - Simple, fast baseline
   - Models should be diverse
   - Works well when models are roughly equal

2. Stacking:
   - When you want best performance
   - Need diverse base Can over models
   -fit if not careful

3. Bagging:
   - Reduce variance
   - Use with unstable models (like decision trees)

4. Boosting:
   - Reduce bias
   - When you need best accuracy
   - More prone to overfitting

Best Practice:
- Start with Voting as baseline
- Move to Stacking for improvement
- Tune individual models first
""")

# ============================================================================
# 12. COMPLETE COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("12. COMPLETE COMPARISON")
print("=" * 70)

# Models to compare
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Voting (Soft)': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('nb', GaussianNB())
        ],
        voting='soft'
    ),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
            ('nb', GaussianNB())
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    )
}

print("\nModel Comparison:")
print("-" * 50)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name:20s}: {acc:.4f}")

# Plot comparison
plt.figure(figsize=(10, 5))
plt.barh(list(results.keys()), list(results.values()), color='steelblue')
plt.xlabel('Accuracy')
plt.title('Ensemble Methods Comparison')
plt.xlim(0.7, 0.95)
for i, (name, acc) in enumerate(results.items()):
    plt.text(acc + 0.005, i, f'{acc:.4f}', va='center')
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STACKING & VOTING SUMMARY")
print("=" * 70)
print("""
Key Takeaways:
1. Voting: Simple majority/average combination
2. Stacking: Learn optimal combination with meta-learner
3. Use diverse base models for best results
4. Stacking often outperforms voting
5. Start simple, then add complexity

Best Practices:
- Tune base models first
- Use cross-validation for stacking
- Try different meta-learners
- Don't overcomplicate if simple works
""")
