"""
=================================================================
04 - STACKING, BLENDING & VOTING ENSEMBLES
=================================================================
Topics covered:
  1. Voting Classifiers (Hard & Soft)
  2. Stacking with sklearn
  3. Manual stacking implementation
  4. Blending
  5. Best practices
=================================================================
"""
import numpy as np
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Dataset ──────────────────────────────────────────────────────
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Section 1: Voting Classifier ─────────────────────────────────
print("=" * 65)
print("SECTION 1: Voting Classifier")
print("=" * 65)

# --- 1.1 Hard Voting ---
print("\n📌 1.1 Hard Voting (majority vote):")
hard_voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(kernel="rbf", random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ],
    voting="hard",
)
hard_voting.fit(X_train, y_train)
print(f"  Hard Voting Accuracy: {accuracy_score(y_test, hard_voting.predict(X_test)):.4f}")

# Individual models
print("\n  Individual model accuracies:")
for name, model in hard_voting.named_estimators_.items():
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"    {name:>5s}: {acc:.4f}")

# --- 1.2 Soft Voting ---
print("\n📌 1.2 Soft Voting (average probabilities):")
soft_voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ],
    voting="soft",
)
soft_voting.fit(X_train, y_train)
print(f"  Soft Voting Accuracy: {accuracy_score(y_test, soft_voting.predict(X_test)):.4f}")

# --- 1.3 Weighted Voting ---
print("\n📌 1.3 Weighted Soft Voting:")
weighted_voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ],
    voting="soft",
    weights=[2, 1, 1],  # RF gets double weight
)
weighted_voting.fit(X_train, y_train)
print(f"  Weighted Voting Accuracy: {accuracy_score(y_test, weighted_voting.predict(X_test)):.4f}")

# ── Section 2: Stacking with sklearn ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Stacking Classifier (sklearn)")
print("=" * 65)

print("\n📌 2.1 Basic Stacking:")
print("""
  Architecture:
    Layer 1 (Base models):
      ├── Random Forest
      ├── Gradient Boosting
      └── KNN
    Layer 2 (Meta-learner):
      └── Logistic Regression
""")

stacking_clf = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,  # 5-fold CV for generating meta-features
)
stacking_clf.fit(X_train, y_train)

print(f"  Stacking Accuracy: {accuracy_score(y_test, stacking_clf.predict(X_test)):.4f}")
print(f"  Stacking CV Score: {cross_val_score(stacking_clf, X_train, y_train, cv=3).mean():.4f}")

# --- 2.2 Stacking with passthrough ---
print("\n📌 2.2 Stacking with passthrough=True:")
print("  (Meta-learner sees base predictions + original features)")
stacking_pass = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=True,  # Include original features
)
stacking_pass.fit(X_train, y_train)
print(f"  With passthrough: {accuracy_score(y_test, stacking_pass.predict(X_test)):.4f}")

# ── Section 3: Manual Stacking ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Manual Stacking Implementation")
print("=" * 65)

from sklearn.model_selection import KFold

print("\n📌 Step-by-step stacking process:")

# Base models
base_models = [
    ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("GB", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
]

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays for meta-features
n_classes = len(np.unique(y_train))
meta_train = np.zeros((X_train.shape[0], len(base_models)))
meta_test = np.zeros((X_test.shape[0], len(base_models)))

print(f"  Creating meta-features using {n_folds}-fold CV...\n")

for i, (name, model) in enumerate(base_models):
    test_preds_folds = np.zeros((X_test.shape[0], n_folds))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr_fold = X_train[train_idx]
        y_tr_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]

        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_tr_fold, y_tr_fold)

        # Out-of-fold predictions → meta-train features
        meta_train[val_idx, i] = model_clone.predict(X_val_fold)
        # Test predictions (average across folds)
        test_preds_folds[:, fold] = model_clone.predict(X_test)

    # Average test predictions across folds
    from scipy import stats
    meta_test[:, i] = stats.mode(test_preds_folds, axis=1)[0].flatten()

    oof_acc = accuracy_score(y_train, meta_train[:, i])
    print(f"  {name}: OOF accuracy = {oof_acc:.4f}")

# Train meta-learner
print(f"\n  Meta-features shape: {meta_train.shape}")
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_train, y_train)
final_pred = meta_model.predict(meta_test)
print(f"  Manual Stacking Accuracy: {accuracy_score(y_test, final_pred):.4f}")

# ── Section 4: Blending ──────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Blending (Simple Stacking)")
print("=" * 65)

print("""
  Blending uses a holdout set instead of cross-validation:
    Training (70%) → train base models
    Holdout  (30%) → get base predictions → train meta-learner
""")

# Split train into train + blend
X_blend_train, X_blend_val, y_blend_train, y_blend_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

blend_meta_val = np.zeros((X_blend_val.shape[0], len(base_models)))
blend_meta_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    model_clone = type(model)(**model.get_params())
    model_clone.fit(X_blend_train, y_blend_train)
    blend_meta_val[:, i] = model_clone.predict(X_blend_val)
    blend_meta_test[:, i] = model_clone.predict(X_test)

blend_meta = LogisticRegression(max_iter=1000)
blend_meta.fit(blend_meta_val, y_blend_val)
blend_pred = blend_meta.predict(blend_meta_test)
print(f"  Blending Accuracy: {accuracy_score(y_test, blend_pred):.4f}")

# ── Section 5: Comparison ────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: All Ensemble Methods Comparison")
print("=" * 65)

print(f"\n  {'Method':<25s} {'Accuracy':>10s}")
print("  " + "-" * 37)
print(f"  {'Hard Voting':<25s} {accuracy_score(y_test, hard_voting.predict(X_test)):>10.4f}")
print(f"  {'Soft Voting':<25s} {accuracy_score(y_test, soft_voting.predict(X_test)):>10.4f}")
print(f"  {'Weighted Voting':<25s} {accuracy_score(y_test, weighted_voting.predict(X_test)):>10.4f}")
print(f"  {'Stacking (sklearn)':<25s} {accuracy_score(y_test, stacking_clf.predict(X_test)):>10.4f}")
print(f"  {'Stacking (manual)':<25s} {accuracy_score(y_test, final_pred):>10.4f}")
print(f"  {'Blending':<25s} {accuracy_score(y_test, blend_pred):>10.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. Soft voting > Hard voting (uses confidence)
  2. Stacking uses CV to avoid overfitting meta-features
  3. Blending is simpler but wastes training data
  4. Use diverse base models for best results
  5. Meta-learner should be simple (LogisticRegression)
  6. passthrough=True can help if features are important

🏗️ Best Practices:
  • Mix different model families (tree, linear, instance-based)
  • Don't stack too many similar models
  • Keep meta-learner simple to avoid overfitting
  • Always cross-validate the full pipeline

📚 Next: exercises.py (Practice Problems)
""")
