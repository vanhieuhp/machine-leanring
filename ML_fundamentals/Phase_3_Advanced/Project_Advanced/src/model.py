"""
Model Module
===========

Contains various ML models for the advanced project.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def get_base_models():
    """Get dictionary of base models"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        )
    }
    return models


def get_ensemble_models():
    """Get ensemble models"""
    models = {
        'Voting (Hard)': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ],
            voting='hard'
        ),
        'Voting (Soft)': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
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
            cv=5
        )
    }
    return models


def train_model(model, X_train, y_train):
    """Train a model"""
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """Make predictions"""
    return model.predict(X_test)


def predict_proba(model, X_test):
    """Get prediction probabilities"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)
    return None


def get_model_by_name(name):
    """Get a specific model by name"""
    base_models = get_base_models()
    ensemble_models = get_ensemble_models()

    all_models = {**base_models, **ensemble_models}

    if name in all_models:
        return all_models[name]

    # Default to Random Forest
    return base_models['Random Forest']


def compare_models(models, X_train, y_train, X_test, y_test):
    """Compare multiple models"""
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': acc
        }

    return results


def tune_random_forest(X_train, y_train, param_grid=None):
    """Tune Random Forest hyperparameters"""
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }

    from sklearn.model_selection import GridSearchCV

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def tune_xgboost(X_train, y_train, param_grid=None):
    """Tune XGBoost hyperparameters"""
    try:
        import xgboost as xgb

        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }

        from sklearn.model_selection import GridSearchCV

        xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        grid_search = GridSearchCV(
            xgb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_

    except ImportError:
        print("XGBoost not available")
        return None, None


if __name__ == "__main__":
    # Test the models
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]

    # Get models
    models = get_base_models()

    # Compare
    results = compare_models(models, X_train, y_train, X_test, y_test)

    print("\nModel Comparison:")
    print("-" * 40)
    for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{name}: {result['accuracy']:.4f}")
