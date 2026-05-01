"""
Kaggle Competitions - Part 4: Ensemble Methods
================================================

This module covers:
- Voting classifiers
- Bagging and Bootstrap
- Boosting algorithms
- Stacking
- Blending
- Model diversity

Based on: Titanic Competition
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, BaggingClassifier,
    StackingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. PREPARING DATA
# ============================================================================

print("=" * 70)
print("1. PREPARING DATA")
print("=" * 70)

# Create sample Titanic-like dataset
np.random.seed(42)
n = 800

data = {
    'Pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.25, 0.55]),
    'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
    'Age': np.random.uniform(1, 80, n),
    'SibSp': np.random.randint(0, 6, n),
    'Parch': np.random.randint(0, 6, n),
    'Fare': np.random.exponential(30, n),
    'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.55, 0.25, 0.20]),
}

df = pd.DataFrame(data)

# Generate survival
def simulate_survival(row):
    prob = 0.3
    if row['Sex'] == 'female':
        prob += 0.4
    if row['Age'] < 16:
        prob += 0.2
    if row['Pclass'] == 1:
        prob += 0.15
    elif row['Pclass'] == 2:
        prob += 0.05
    if row['SibSp'] + row['Parch'] > 3:
        prob -= 0.1
    return 1 if np.random.random() < prob else 0

df['Survived'] = df.apply(simulate_survival, axis=1)

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = np.random.choice(['Mr', 'Miss', 'Mrs', 'Master'], n)
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])

# Encode
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3})
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['AgeBin'] = df['AgeBin'].astype(int)

feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title', 'AgeBin']

X = df[feature_cols]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {len(feature_cols)}")

# ============================================================================
# 2. VOTING ENSEMBLES
# ============================================================================

print("\n" + "=" * 70)
print("2. VOTING ENSEMBLES")
print("=" * 70)

# -------------------------------------------------------------------------
# 2.1 Hard Voting
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.1 Hard Voting (Majority Vote)")
print("-" * 50)

# Define base models
model1 = LogisticRegression(random_state=42, max_iter=1000)
model2 = DecisionTreeClassifier(random_state=42)
model3 = KNeighborsClassifier(n_neighbors=5)

# Create voting classifier
voting_hard = VotingClassifier(
    estimators=[
        ('lr', model1),
        ('dt', model2),
        ('knn', model3)
    ],
    voting='hard'
)

# Train
voting_hard.fit(X_train, y_train)

# Evaluate
y_pred = voting_hard.predict(X_test)
hard_vote_score = accuracy_score(y_test, y_pred)
hard_vote_f1 = f1_score(y_test, y_pred)

print(f"Hard Voting Accuracy: {hard_vote_score:.4f}")
print(f"Hard Voting F1: {hard_vote_f1:.4f}")

# Compare with individual models
for name, model in [('Logistic', model1), ('Decision Tree', model2), ('KNN', model3)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  {name}: Accuracy={accuracy_score(y_test, pred):.4f}")

# -------------------------------------------------------------------------
# 2.2 Soft Voting
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.2 Soft Voting (Probability Average)")
print("-" * 50)

# Models that support predict_proba
model1 = LogisticRegression(random_state=42, max_iter=1000)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model3 = GradientBoostingClassifier(n_estimators=100, random_state=42)

voting_soft = VotingClassifier(
    estimators=[
        ('lr', model1),
        ('rf', model2),
        ('gb', model3)
    ],
    voting='soft'
)

voting_soft.fit(X_train, y_train)

y_pred = voting_soft.predict(X_test)
y_pred_proba = voting_soft.predict_proba(X_test)[:, 1]

soft_vote_score = accuracy_score(y_test, y_pred)
soft_vote_f1 = f1_score(y_test, y_pred)
soft_vote_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Soft Voting Accuracy: {soft_vote_score:.4f}")
print(f"Soft Voting F1: {soft_vote_f1:.4f}")
print(f"Soft Voting ROC-AUC: {soft_vote_auc:.4f}")

# -------------------------------------------------------------------------
# 2.3 Weighted Voting
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.3 Weighted Voting")
print("-" * 50)

# Give more weight to better models
voting_weighted = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ],
    voting='soft',
    weights=[1, 2, 3]  # Give more weight to boosting
)

voting_weighted.fit(X_train, y_train)

y_pred = voting_weighted.predict(X_test)
weighted_score = accuracy_score(y_test, y_pred)
print(f"Weighted Voting Accuracy: {weighted_score:.4f}")

# ============================================================================
# 3. BAGGING
# ============================================================================

print("\n" + "=" * 70)
print("3. BAGGING (Bootstrap Aggregating)")
print("=" * 70)

# -------------------------------------------------------------------------
# 3.1 Bagging Classifier
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.1 Bagging with Decision Tree")
print("-" * 50)

# Bagging with Decision Tree base
bagging_dt = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.8,  # Use 80% of samples per bag
    max_features=0.8,  # Use 80% of features per bag
    random_state=42,
    n_jobs=-1
)

bagging_dt.fit(X_train, y_train)
y_pred = bagging_dt.predict(X_test)
bagging_score = accuracy_score(y_test, y_pred)

print(f"Bagging (DT) Accuracy: {bagging_score:.4f}")

# -------------------------------------------------------------------------
# 3.2 Random Forest
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.2 Random Forest (Special case of Bagging)")
print("-" * 50)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy: {rf_score:.4f}")
print(f"Number of trees: {rf.n_estimators}")
print(f"Number of features: {rf.n_features_in_}")

# -------------------------------------------------------------------------
# 3.3 Extra Trees
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.3 Extra Trees (Extremely Randomized Trees)")
print("-" * 50)

et = ExtraTreesClassifier(n_estimators=200, random_state=42)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
et_score = accuracy_score(y_test, y_pred)

print(f"Extra Trees Accuracy: {et_score:.4f}")

# Compare
print("\nBagging Methods Comparison:")
print(f"  Bagging (DT): {bagging_score:.4f}")
print(f"  Random Forest: {rf_score:.4f}")
print(f"  Extra Trees: {et_score:.4f}")

# ============================================================================
# 4. BOOSTING
# ============================================================================

print("\n" + "=" * 70)
print("4. BOOSTING (Sequential Ensemble)")
print("=" * 70)

# -------------------------------------------------------------------------
# 4.1 AdaBoost
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.1 AdaBoost (Adaptive Boosting)")
print("-" * 50)

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
ada_score = accuracy_score(y_test, y_pred)

print(f"AdaBoost Accuracy: {ada_score:.4f}")
print(f"Number of estimators: {adaboost.n_estimators}")

# -------------------------------------------------------------------------
# 4.2 Gradient Boosting
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.2 Gradient Boosting")
print("-" * 50)

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
gb_score = accuracy_score(y_test, y_pred)

print(f"Gradient Boosting Accuracy: {gb_score:.4f}")

# Feature importance
print("\nFeature Importance (GB):")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.head(5).iterrows():
    bar = '█' * int(row['importance'] * 40)
    print(f"  {row['feature']:15s}: {row['importance']:.4f} {bar}")

# -------------------------------------------------------------------------
# 4.3 XGBoost
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.3 XGBoost (eXtreme Gradient Boosting)")
print("-" * 50)

try:
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    xgb_score = accuracy_score(y_test, y_pred)

    print(f"XGBoost Accuracy: {xgb_score:.4f}")

except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    xgb_score = 0

# -------------------------------------------------------------------------
# 4.4 LightGBM
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.4 LightGBM (Light Gradient Boosting)")
print("-" * 50)

try:
    import lightgbm as lgb

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=-1
    )

    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    lgb_score = accuracy_score(y_test, y_pred)

    print(f"LightGBM Accuracy: {lgb_score:.4f}")

except ImportError:
    print("LightGBM not installed. Install with: pip install lightgbm")
    lgb_score = 0

# -------------------------------------------------------------------------
# 4.5 CatBoost
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.5 CatBoost (Categorical Boosting)")
print("-" * 50)

try:
    from catboost import CatBoostClassifier

    cat_model = CatBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )

    cat_model.fit(X_train, y_train)
    y_pred = cat_model.predict(X_test)
    cat_score = accuracy_score(y_test, y_pred)

    print(f"CatBoost Accuracy: {cat_score:.4f}")

except ImportError:
    print("CatBoost not installed. Install with: pip install catboost")
    cat_score = 0

# Summary
print("\nBoosting Methods Comparison:")
print(f"  AdaBoost: {ada_score:.4f}")
print(f"  Gradient Boosting: {gb_score:.4f}")
if xgb_score > 0:
    print(f"  XGBoost: {xgb_score:.4f}")
if lgb_score > 0:
    print(f"  LightGBM: {lgb_score:.4f}")
if cat_score > 0:
    print(f"  CatBoost: {cat_score:.4f}")

# ============================================================================
# 5. STACKING
# ============================================================================

print("\n" + "=" * 70)
print("5. STACKING (Stacked Generalization)")
print("=" * 70)

# -------------------------------------------------------------------------
# 5.1 Basic Stacking
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.1 Basic Stacking")
print("-" * 50)

# Define base models
base_models = [
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
]

# Meta-learner
meta_learner = LogisticRegression(random_state=42)

# Create stacking classifier
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,  # Use cross-validation for OOF predictions
    passthrough=False  # Only use OOF predictions as meta-features
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
stacking_score = accuracy_score(y_test, y_pred)

print(f"Stacking Accuracy: {stacking_score:.4f}")

# -------------------------------------------------------------------------
# 5.2 Stacking with Passthrough
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.2 Stacking with Original Features")
print("-" * 50)

stacking_passthrough = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    passthrough=True  # Include original features in meta-features
)

stacking_passthrough.fit(X_train, y_train)
y_pred = stacking_passthrough.predict(X_test)
stacking_pt_score = accuracy_score(y_test, y_pred)

print(f"Stacking (passthrough) Accuracy: {stacking_pt_score:.4f}")

# ============================================================================
# 6. BLENDING
# ============================================================================

print("\n" + "=" * 70)
print("6. BLENDING (Simplified Stacking)")
print("=" * 70)

print("\n" + "-" * 50)
print("6.1 Blending Implementation")
print("-" * 50)

# Manual blending implementation
def blend_predictions(models, X_train, y_train, X_test, holdout_ratio=0.2, seed=42):
    """
    Simple blending implementation.

    1. Split training data into train and holdout
    2. Train each base model on train portion
    3. Get predictions on holdout and test set
    4. Train meta-learner on holdout predictions
    5. Generate final predictions on test set
    """

    # Split into train and holdout
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X_train, y_train, test_size=holdout_ratio, random_state=seed, stratify=y_train
    )

    # Get OOF predictions for holdout
    holdout_preds = np.zeros((len(X_hold), len(models)))
    test_preds = np.zeros((len(X_test), len(models)))

    for i, (name, model) in enumerate(models):
        # Clone model
        model_clone = model.__class__(**model.get_params())

        # Train on train portion
        model_clone.fit(X_tr, y_tr)

        # Predict on holdout (use predict_proba for soft voting)
        if hasattr(model_clone, 'predict_proba'):
            holdout_preds[:, i] = model_clone.predict_proba(X_hold)[:, 1]
            test_preds[:, i] = model_clone.predict_proba(X_test)[:, 1]
        else:
            holdout_preds[:, i] = model_clone.predict(X_hold)
            test_preds[:, i] = model_clone.predict(X_test)

    # Train meta-learner on holdout predictions
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(holdout_preds, y_hold)

    # Final predictions
    final_preds = meta_model.predict(test_preds)
    final_proba = meta_model.predict_proba(test_preds)[:, 1]

    return final_preds, final_proba, meta_model

# Define models
blend_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
]

# Blend
y_pred_blend, y_proba_blend, meta_model = blend_predictions(blend_models, X_train, y_train, X_test)

blend_score = accuracy_score(y_test, y_pred_blend)
blend_auc = roc_auc_score(y_test, y_proba_blend)

print(f"Blending Accuracy: {blend_score:.4f}")
print(f"Blending ROC-AUC: {blend_auc:.4f}")

# ============================================================================
# 7. ADVANCED ENSEMBLE TECHNIQUES
# ============================================================================

print("\n" + "=" * 70)
print("7. ADVANCED ENSEMBLE TECHNIQUES")
print("=" * 70)

# -------------------------------------------------------------------------
# 7.1 Model Diversity
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("7.1 Creating Diverse Ensemble")
print("-" * 50)

# Train diverse models with different seeds
diverse_models = []
for seed in [42, 123, 456]:
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=7)
    rf.fit(X_train, y_train)
    diverse_models.append(rf)

# Average predictions
preds = np.array([m.predict_proba(X_test)[:, 1] for m in diverse_models])
avg_proba = preds.mean(axis=0)
avg_pred = (avg_proba > 0.5).astype(int)

diverse_score = accuracy_score(y_test, avg_pred)
print(f"Diverse Models (3 RFs) Accuracy: {diverse_score:.4f}")

# -------------------------------------------------------------------------
# 7.2 Rank Averaging
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("7.2 Rank Averaging")
print("-" * 50)

# Get predictions from multiple models
models_for_rank = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    LogisticRegression(random_state=42, max_iter=1000)
]

all_preds = []
for m in models_for_rank:
    m.fit(X_train, y_train)
    all_preds.append(m.predict_proba(X_test)[:, 1])

# Convert to ranks and average
ranks = np.array([pd.Series(p).rank() for p in all_preds])
avg_rank = ranks.mean(axis=0)
rank_pred = (avg_rank > len(X_test) / 2).astype(int)

rank_score = accuracy_score(y_test, rank_pred)
print(f"Rank Averaging Accuracy: {rank_score:.4f}")

# ============================================================================
# 8. COMPLETE ENSEMBLE PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("8. COMPLETE ENSEMBLE PIPELINE")
print("=" * 70)

def create_ensemble(X_train, y_train, X_test, y_test):
    """
    Complete ensemble pipeline with multiple strategies.
    """

    results = {}

    # 1. Individual models
    print("\n1. Individual Models:")
    individual_models = {
        'LR': LogisticRegression(random_state=42, max_iter=1000),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    for name, model in individual_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        results[name] = score
        print(f"  {name}: {score:.4f}")

    # 2. Voting
    print("\n2. Voting Ensembles:")
    voting = VotingClassifier(
        estimators=list(individual_models.items()),
        voting='soft'
    )
    voting.fit(X_train, y_train)
    pred = voting.predict(X_test)
    score = accuracy_score(y_test, pred)
    results['Voting'] = score
    print(f"  Soft Voting: {score:.4f}")

    # 3. Stacking
    print("\n3. Stacking:")
    stacking = StackingClassifier(
        estimators=list(individual_models.items()),
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    stacking.fit(X_train, y_train)
    pred = stacking.predict(X_test)
    score = accuracy_score(y_test, pred)
    results['Stacking'] = score
    print(f"  Stacking: {score:.4f}")

    # 4. Blending
    print("\n4. Blending:")
    blend_pred, _, _ = blend_predictions(
        list(individual_models.items()), X_train, y_train, X_test
    )
    score = accuracy_score(y_test, blend_pred)
    results['Blending'] = score
    print(f"  Blending: {score:.4f}")

    # Summary
    print("\n" + "-" * 50)
    print("FINAL RESULTS:")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")

    return results

results = create_ensemble(X_train, y_train, X_test, y_test)

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Voting Ensembles
   - Hard voting: Majority vote
   - Soft voting: Average probabilities (better when models have predict_proba)
   - Weighted voting: Give more weight to better models

2. Bagging
   - Reduces variance
   - Random Forest: Bagging + feature sampling
   - Extra Trees: More randomization than RF

3. Boosting
   - Reduces bias
   - Sequential: Each model learns from previous errors
   - Popular: XGBoost, LightGBM, CatBoost

4. Stacking
   - Train meta-learner on base model predictions
   - Use OOF predictions to avoid leakage
   - Can include original features (passthrough)

5. Blending
   - Similar to stacking but uses holdout set
   - Simpler but less robust than stacking

6. Best Practices
   - Use diverse base models
   - Don't overcomplicate if simple voting works
   - Always validate with cross-validation

Next: exercises.py to practice these techniques
""")
