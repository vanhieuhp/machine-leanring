"""
Kaggle Competitions - Part 3: Model Training & Hyperparameter Tuning
====================================================================

This module covers:
- Training various ML models
- Cross-validation strategies
- Hyperparameter optimization
- Model evaluation metrics
- Model persistence

Based on: Titanic Competition
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
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

# Generate synthetic Titanic data
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

# Generate survival (based on known patterns)
def simulate_survival(row):
    prob = 0.3  # base probability

    # Women and children first
    if row['Sex'] == 'female':
        prob += 0.4
    if row['Age'] < 16:
        prob += 0.2

    # Class matters
    if row['Pclass'] == 1:
        prob += 0.15
    elif row['Pclass'] == 2:
        prob += 0.05

    # Family
    if row['SibSp'] + row['Parch'] > 3:
        prob -= 0.1

    return 1 if np.random.random() < prob else 0

df['Survived'] = df.apply(simulate_survival, axis=1)

print(f"Dataset shape: {df.shape}")
print(f"Survival rate: {df['Survived'].mean()*100:.2f}%")
print(f"\nClass distribution:")
print(df['Survived'].value_counts())

# -------------------------------------------------------------------------
# 1.1 Feature Engineering
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.1 Feature Engineering")
print("-" * 50)

# Create features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = np.random.choice(['Mr', 'Miss', 'Mrs', 'Master'], n)
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3})

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['AgeBin'] = df['AgeBin'].astype(int)

# Select features
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title', 'AgeBin']

X = df[feature_cols]
y = df['Survived']

print(f"Features: {feature_cols}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# -------------------------------------------------------------------------
# 1.2 Train-Test Split
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.2 Train-Test Split")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training survival rate: {y_train.mean()*100:.2f}%")
print(f"Test survival rate: {y_test.mean()*100:.2f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 2. BASELINE MODELS
# ============================================================================

print("\n" + "=" * 70)
print("2. BASELINE MODELS")
print("=" * 70)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, random_state=42),
}

# Train and evaluate each model
results = []

for name, model in models.items():
    # Use scaled data for SVM and KNN, unscaled for tree-based
    if name in ['SVM', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc
    })

    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

# Results summary
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)
print(results_df.sort_values('F1', ascending=False).to_string(index=False))

# ============================================================================
# 3. CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("3. CROSS-VALIDATION")
print("=" * 70)

# -------------------------------------------------------------------------
# 3.1 K-Fold Cross-Validation
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.1 K-Fold Cross-Validation")
print("-" * 50)

# Using cross_val_score
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold CV
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("5-Fold Cross-Validation:")
print(f"  Scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f}")
print(f"  Std: {cv_scores.std():.4f}")

# -------------------------------------------------------------------------
# 3.2 Stratified K-Fold
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.2 Stratified K-Fold (for imbalanced data)")
print("-" * 50)

# For imbalanced data, use stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_stratified = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_fold, y_train_fold)

    y_pred = model.predict(X_val_fold)
    score = accuracy_score(y_val_fold, y_pred)
    cv_scores_stratified.append(score)

    print(f"  Fold {fold+1}: {score:.4f}")

print(f"\nMean: {np.mean(cv_scores_stratified):.4f}")
print(f"Std: {np.std(cv_scores_stratified):.4f}")

# -------------------------------------------------------------------------
# 3.3 Cross-Validation with Multiple Metrics
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.3 CV with Multiple Metrics")
print("-" * 50)

from sklearn.model_selection import cross_validate

model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
    return_train_score=True
)

print("Cross-validation results:")
for metric in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
    test_key = f'test_{metric}'
    print(f"  {metric}: {cv_results[test_key].mean():.4f} (+/- {cv_results[test_key].std():.4f})")

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "=" * 70)
print("4. HYPERPARAMETER TUNING")
print("=" * 70)

# -------------------------------------------------------------------------
# 4.1 Grid Search
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.1 Grid Search CV")
print("-" * 50)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# For demonstration, use a smaller grid
param_grid_small = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
}

print(f"Total combinations: {np.prod([len(v) for v in param_grid_small.values()])}")

# GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf,
    param_grid_small,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test F1 score: {f1_score(y_test, y_pred):.4f}")

# -------------------------------------------------------------------------
# 4.2 Randomized Search
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.2 Randomized Search CV")
print("-" * 50)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf,
    param_dist,
    n_iter=20,  # Number of random combinations to try
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")

# Evaluate on test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test F1 score: {f1_score(y_test, y_pred):.4f}")

# -------------------------------------------------------------------------
# 4.3 Optuna (Bayesian Optimization)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.3 Optuna (Bayesian Optimization)")
print("-" * 50)

try:
    import optuna
    from optuna.samplers import TPESampler

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        }

        model = GradientBoostingClassifier(random_state=42, **params)

        cv = StratifiedKFold(n_splits=3, shuffle=True)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

        return scores.mean()

    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    print(f"Best trial value: {study.best_trial.value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")

    # Train with best params
    best_model = GradientBoostingClassifier(random_state=42, **study.best_trial.params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Test F1 score: {f1_score(y_test, y_pred):.4f}")

except ImportError:
    print("Optuna not installed. Install with: pip install optuna")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("5. MODEL EVALUATION")
print("=" * 70)

# Train best model
best_rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# -------------------------------------------------------------------------
# 5.1 Confusion Matrix
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.1 Confusion Matrix")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# -------------------------------------------------------------------------
# 5.2 Classification Report
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.2 Classification Report")
print("-" * 50)

print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# -------------------------------------------------------------------------
# 5.3 ROC Curve
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.3 ROC Curve Analysis")
print("-" * 50)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Optimal threshold: {thresholds[np.argmax(tpr - fpr)]:.4f}")

# Show some threshold points
print("\nThreshold analysis:")
for i in [0, len(thresholds)//4, len(thresholds)//2, len(thresholds)*3//4, -1]:
    if i < len(thresholds):
        print(f"  Threshold {thresholds[i]:.2f}: TPR={tpr[i]:.3f}, FPR={fpr[i]:.3f}")

# -------------------------------------------------------------------------
# 5.4 Feature Importance
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.4 Feature Importance")
print("-" * 50)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
for _, row in feature_importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:15s}: {row['importance']:.4f} {bar}")

# ============================================================================
# 6. MODEL PERSISTENCE
# ============================================================================

print("\n" + "=" * 70)
print("6. MODEL PERSISTENCE")
print("=" * 70)

# -------------------------------------------------------------------------
# 6.1 Save Model with Pickle
# -------------------------------------------------------------------------

import pickle
import os

# Save model
model_path = 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_rf, f)

print(f"Model saved to: {model_path}")

# Load model
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Verify
y_pred_loaded = loaded_model.predict(X_test)
print(f"Loaded model accuracy: {accuracy_score(y_test, y_pred_loaded):.4f}")

# Clean up
os.remove(model_path)

# -------------------------------------------------------------------------
# 6.2 Save Model with Joblib
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("6.2 Save Model with Joblib (better for large models)")
print("-" * 50)

try:
    import joblib

    # Save
    joblib_path = 'best_model.joblib'
    joblib.dump(best_rf, joblib_path)
    print(f"Model saved to: {joblib_path}")

    # Load
    loaded_model = joblib.load(joblib_path)
    y_pred_loaded = loaded_model.predict(X_test)
    print(f"Loaded model accuracy: {accuracy_score(y_test, y_pred_loaded):.4f}")

    os.remove(joblib_path)

except ImportError:
    print("Joblib not installed. Install with: pip install joblib")

# -------------------------------------------------------------------------
# 6.3 Save Scaler
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("6.3 Save Preprocessing Objects")
print("-" * 50)

# Save scaler and model together
preprocessing_pipeline = {
    'scaler': scaler,
    'model': best_rf,
    'feature_cols': feature_cols
}

# In practice, save to file
print("Preprocessing pipeline ready:")
print(f"  Scaler: {type(scaler).__name__}")
print(f"  Model: {type(best_rf).__name__}")
print(f"  Features: {feature_cols}")

# ============================================================================
# 7. COMPLETE TRAINING PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("7. COMPLETE TRAINING PIPELINE")
print("=" * 70)

def train_model(X, y, model_type='rf', tuning='grid', cv_folds=5):
    """
    Complete training pipeline with cross-validation and hyperparameter tuning.

    Parameters:
    -----------
    X : DataFrame - Feature matrix
    y : Series - Target variable
    model_type : str - 'rf', 'gb', 'xgb', or 'lr'
    tuning : str - 'grid', 'random', or 'none'
    cv_folds : int - Number of CV folds

    Returns:
    --------
    best_model, results
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    models = {
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
    }

    model = models[model_type]

    # Hyperparameter tuning
    if tuning == 'grid':
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, None],
                'min_samples_split': [2, 5],
            }
        elif model_type == 'gb':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
            }
        else:
            param_grid = {'C': [0.1, 1, 10]}

        search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1)

    elif tuning == 'random':
        param_dist = {
            'n_estimators': randint(50, 300),
            'max_depth': [3, 5, 7, None],
        }
        search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=cv_folds,
                                     scoring='f1', n_jobs=-1, random_state=42)
    else:
        search = model

    # Train
    search.fit(X_train, y_train)

    # Get best model
    if hasattr(search, 'best_estimator_'):
        best_model = search.best_estimator_
    else:
        best_model = search

    # Evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    results = {
        'model': best_model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
    }

    if tuning in ['grid', 'random']:
        results['best_params'] = search.best_params_
        results['cv_score'] = search.best_score_

    return best_model, results

# Run pipeline
print("Running complete training pipeline...")
model, results = train_model(X, y, model_type='rf', tuning='grid')

print("\nResults:")
for key, value in results.items():
    if key != 'model':
        print(f"  {key}: {value}")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Start with Baseline
   - Train simple models first
   - Establish baseline performance
   - Then try more complex models

2. Cross-Validation
   - Always use stratified k-fold for classification
   - 5-fold is a good default
   - Trust CV over single train/test split

3. Hyperparameter Tuning
   - GridSearch: Exhaustive, slow
   - RandomizedSearch: Faster, good for large spaces
   - Optuna: Bayesian, most efficient for complex spaces

4. Evaluation Metrics
   - Accuracy: Overall correctness
   - Precision: Of predicted positive, how many are correct
   - Recall: Of actual positive, how many we found
   - F1: Harmonic mean of precision and recall
   - ROC-AUC: Threshold-independent measure

5. Model Selection
   - Compare multiple models
   - Consider speed vs accuracy trade-off
   - Check feature importance for interpretability

6. Save Everything
   - Save models and preprocessing
   - Document parameters and CV scores
   - Reproducibility is key

Next: Ensembling (04_ensembling.py)
""")
