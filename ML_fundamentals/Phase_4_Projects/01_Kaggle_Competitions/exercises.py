"""
Kaggle Competitions - Practice Exercises
========================================

Complete these exercises to solidify your Kaggle competition skills.
Solutions are provided at the bottom.

Based on: Titanic Competition Workflow
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXERCISE 1: Data Loading and Exploration
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Data Loading and Exploration")
print("=" * 70)

# 1.1 Create sample Titanic dataset
# TODO: Create a DataFrame with the following columns:
# PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# Generate 100 samples with realistic patterns
# HINT: Use np.random.choice for categorical, np.random for numerical

np.random.seed(42)
n = 100

data = {
    'PassengerId': range(1, n + 1),
    'Pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.25, 0.55]),
    'Name': [f'Passenger_{i}' for i in range(1, n + 1)],
    'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
    'Age': np.random.uniform(1, 80, n),
    'SibSp': np.random.randint(0, 6, n),
    'Parch': np.random.randint(0, 6, n),
    'Ticket': [f'T{i:06d}' for i in range(1, n + 1)],
    'Fare': np.random.exponential(30, n),
    'Cabin': [np.nan] * n,  # Most cabins are missing
    'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.55, 0.25, 0.20])
}

df = pd.DataFrame(data)

# TODO: Add some cabin values (about 20%)
cabin_indices = np.random.choice(n, 20, replace=False)
deck_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for idx in cabin_indices:
    deck = np.random.choice(deck_letters)
    cabin_num = np.random.randint(1, 50)
    df.loc[idx, 'Cabin'] = f'{deck}{cabin_num}'

# TODO: Add some missing ages (about 15%)
age_indices = np.random.choice(n, 15, replace=False)
df.loc[age_indices, 'Age'] = np.nan

# TODO: Add a Target column 'Survived' based on patterns
# - Females have higher survival rate
# - Higher class has higher survival rate
# - Children (<16) have higher survival rate

def get_survival(row):
    prob = 0.3
    if row['Sex'] == 'female':
        prob += 0.4
    if row['Age'] < 16:
        prob += 0.2
    if row['Pclass'] == 1:
        prob += 0.15
    elif row['Pclass'] == 2:
        prob += 0.05
    return 1 if np.random.random() < prob else 0

df['Survived'] = df.apply(get_survival, axis=1)

# 1.2 Display basic info
# TODO: Print the shape and first few rows
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 1.3 Check missing values
# TODO: Print missing values count for each column
print("\nMissing values:")
print(df.isnull().sum())

# 1.4 Check data types
# TODO: Print data types
print("\nData types:")
print(df.dtypes)

# ============================================================================
# EXERCISE 2: Feature Engineering
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Feature Engineering")
print("=" * 70)

# Using the df from Exercise 1

# 2.1 Extract Title from Name
# TODO: Create a Title column by extracting from Name
# Hint: Look for patterns like "Mr.", "Mrs.", "Miss", "Master"

def extract_title(name):
    """Extract title from name."""
    # TODO: Implement
    return 'Other'

df['Title'] = df['Name'].apply(extract_title)
print("Title distribution:")
print(df['Title'].value_counts())

# 2.2 Create FamilySize feature
# TODO: Create FamilySize = SibSp + Parch + 1
df['FamilySize'] = None  # TODO: Calculate

# 2.3 Create IsAlone feature
# TODO: Create IsAlone = 1 if FamilySize == 1, else 0
df['IsAlone'] = None  # TODO: Calculate

print("\nFamily features:")
print(df[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head())

# 2.4 Extract Deck from Cabin
# TODO: Create Deck column (first letter of Cabin, 'U' for Unknown)
df['Deck'] = None  # TODO: Calculate

# 2.5 Handle missing Age values
# TODO: Fill missing Age with median age
median_age = None  # TODO: Calculate
df['Age'] = None  # TODO: Fill

print("\nAge after imputation:")
print(df['Age'].describe())

# 2.6 Encode categorical variables
# TODO: Encode 'Sex', 'Embarked', 'Title'
# Sex: male=0, female=1
# Embarked: S=0, C=1, Q=2
# Title: Use LabelEncoder or manual mapping

df['Sex_encoded'] = None  # TODO: Encode
df['Embarked_encoded'] = None  # TODO: Encode

print("\nEncoded categorical variables:")
print(df[['Sex', 'Sex_encoded', 'Embarked', 'Embarked_encoded']].head())

# ============================================================================
# EXERCISE 3: Model Training
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Model Training")
print("=" * 70)

# Prepare features
feature_cols = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked_encoded', 'FamilySize', 'IsAlone']

X = df[feature_cols].copy()
y = df['Survived']

# Fill any remaining NaN
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3.1 Train a Decision Tree
# TODO: Train DecisionTreeClassifier and evaluate
dt_model = None  # TODO: Create and train
dt_pred = None  # TODO: Predict
dt_accuracy = None  # TODO: Calculate

print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# 3.2 Train a Random Forest
# TODO: Train RandomForestClassifier with n_estimators=100
rf_model = None  # TODO: Create and train
rf_pred = None  # TODO: Predict
rf_accuracy = None  # TODO: Calculate

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# 3.3 Train a Logistic Regression
# TODO: Train LogisticRegression (remember to scale features!)
scaler = StandardScaler()
X_train_scaled = None  # TODO: Fit and transform
X_test_scaled = None  # TODO: Transform

lr_model = None  # TODO: Create and train
lr_pred = None  # TODO: Predict
lr_accuracy = None  # TODO: Calculate

print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# ============================================================================
# EXERCISE 4: Cross-Validation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Cross-Validation")
print("=" * 70)

# 4.1 5-Fold Cross-Validation
# TODO: Use cross_val_score with cv=5 for RandomForest
cv_scores = None  # TODO: Calculate

print("5-Fold CV Scores:", cv_scores)
print(f"Mean: {cv_scores.mean():.4f}")
print(f"Std: {cv_scores.std():.4f}")

# 4.2 Stratified K-Fold
# TODO: Use StratifiedKFold for imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # TODO: Train and evaluate
    pass

print(f"\nStratified K-Fold Scores: {stratified_scores}")
print(f"Mean: {np.mean(stratified_scores):.4f}")

# ============================================================================
# EXERCISE 5: Hyperparameter Tuning
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Hyperparameter Tuning")
print("=" * 70)

# 5.1 Grid Search for Decision Tree
# TODO: Use GridSearchCV to find best max_depth and min_samples_split
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

dt_gs = None  # TODO: Create GridSearchCV
dt_gs.fit(X_train, y_train)

print(f"Best parameters: {dt_gs.best_params_}")
print(f"Best CV score: {dt_gs.best_score_:.4f}")

# 5.2 Random Search for Random Forest
# TODO: Use RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, None],
    'min_samples_split': randint(2, 10),
}

rf_rs = None  # TODO: Create RandomizedSearchCV
rf_rs.fit(X_train, y_train)

print(f"\nBest parameters: {rf_rs.best_params_}")
print(f"Best CV score: {rf_rs.best_score_:.4f}")

# ============================================================================
# EXERCISE 6: Ensemble Methods
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Ensemble Methods")
print("=" * 70)

# 6.1 Voting Classifier
# TODO: Create VotingClassifier with 'soft' voting using LR, RF, GB

voting_clf = None  # TODO: Create
voting_clf.fit(X_train, y_train)
voting_pred = None  # TODO: Predict
voting_accuracy = None  # TODO: Calculate

print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")

# 6.2 Manual Averaging
# TODO: Train LR, RF, GB separately, get probabilities, average them

# Train models
lr = LogisticRegression(random_state=42, max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Get probabilities
lr_proba = None  # TODO: Get probabilities
rf_proba = None  # TODO: Get probabilities
gb_proba = None  # TODO: Get probabilities

# Average
avg_proba = None  # TODO: Average
avg_pred = None  # TODO: Convert to binary

manual_accuracy = None  # TODO: Calculate

print(f"Manual Ensemble Accuracy: {manual_accuracy:.4f}")

# ============================================================================
# EXERCISE 7: Model Evaluation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Model Evaluation")
print("=" * 70)

# Use the best model from voting classifier
best_model = voting_clf
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 7.1 Accuracy
# TODO: Calculate accuracy
accuracy = None  # TODO: Calculate

# 7.2 Precision, Recall, F1
# TODO: Calculate precision, recall, f1
precision = None  # TODO: Calculate
recall = None  # TODO: Calculate
f1 = None  # TODO: Calculate

# 7.3 ROC-AUC
# TODO: Calculate ROC-AUC
roc_auc = None  # TODO: Calculate

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# 7.4 Classification Report
# TODO: Print classification report
print("\nClassification Report:")
print("TODO: Add classification_report")

# ============================================================================
# EXERCISE 8: Feature Importance
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Feature Importance")
print("=" * 70)

# Train Random Forest
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X_train, y_train)

# Get feature importance
# TODO: Get feature importances and sort
importance = None  # TODO: Get

print("Feature Importance:")
for feat, imp in importance:
    bar = '█' * int(imp * 50)
    print(f"  {feat:20s}: {imp:.4f} {bar}")

# ============================================================================
# EXERCISE 9: Complete Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9: Complete Pipeline")
print("=" * 70)

def full_pipeline(df, test_size=0.2):
    """
    Complete ML pipeline from raw data to predictions.
    """
    # 1. Feature Engineering
    # TODO: Implement feature engineering

    # 2. Prepare features
    # TODO: Select features

    # 3. Split data
    # TODO: Train-test split

    # 4. Train model
    # TODO: Train ensemble

    # 5. Evaluate
    # TODO: Return metrics

    return {'accuracy': 0, 'f1': 0, 'roc_auc': 0}

# Run pipeline
# results = full_pipeline(df)
# print(f"\nFinal Results: {results}")

# ============================================================================
# EXERCISE 10: Kaggle Submission Format
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 10: Kaggle Submission Format")
print("=" * 70)

# Create sample submission file
# TODO: Create submission DataFrame with PassengerId and Survived columns
submission = None  # TODO: Create

print("Submission format:")
print(submission.head())

# TODO: Save to CSV (in practice)
# submission.to_csv('submission.csv', index=False)

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("""
EXERCISE 1:
- Dataset created with 100 samples
- Missing values in Age (~15%) and Cabin (~80%)

EXERCISE 2:
2.1 Title extraction:
    def extract_title(name):
        if 'Mr.' in name: return 'Mr'
        elif 'Mrs.' in name: return 'Mrs'
        elif 'Miss' in name: return 'Miss'
        elif 'Master' in name: return 'Master'
        else: return 'Other'

2.2 FamilySize: df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
2.3 IsAlone: df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
2.4 Deck: df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
2.5 Fill Age: df['Age'].fillna(df['Age'].median(), inplace=True)
2.6 Encoding: Use map() or LabelEncoder

EXERCISE 3:
3.1: dt_model = DecisionTreeClassifier(random_state=42)
3.2: rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
3.3: scaler = StandardScaler(); lr = LogisticRegression()

EXERCISE 4:
4.1: cross_val_score(RandomForestClassifier(), X, y, cv=5)
4.2: StratifiedKFold with n_splits=5

EXERCISE 5:
5.1: GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3)
5.2: RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=10)

EXERCISE 6:
6.1: VotingClassifier with voting='soft'
6.2: Average predict_proba from multiple models

EXERCISE 7:
- accuracy_score(y_test, y_pred)
- precision_score(y_test, y_pred)
- recall_score(y_test, y_pred)
- f1_score(y_test, y_pred)
- roc_auc_score(y_test, y_pred_proba)

EXERCISE 8:
- rf.feature_importances_

EXERCISE 9:
- Combine all steps in full_pipeline function

EXERCISE 10:
- submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
""")
