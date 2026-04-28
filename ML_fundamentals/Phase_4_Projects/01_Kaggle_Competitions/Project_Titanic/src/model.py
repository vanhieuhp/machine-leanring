"""
Model Module
===========

Defines machine learning models for Titanic prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def get_baseline_models():
    """
    Get dictionary of baseline models.

    Returns:
    --------
    dict
        Dictionary of baseline models
    """
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        ),
        'DecisionTree': DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=3,
            learning_rate=0.1
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'NaiveBayes': GaussianNB()
    }

    return models


def get_advanced_models():
    """
    Get dictionary of advanced models (with tuning).

    Returns:
    --------
    dict
        Dictionary of advanced models
    """
    models = {
        'RandomForestTuned': RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'GradientBoostingTuned': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        )
    }

    # Try to add XGBoost, LightGBM, CatBoost if available
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            verbose=0
        )
    except ImportError:
        pass

    return models


def get_voting_ensemble(models_dict, voting='soft', weights=None):
    """
    Create voting ensemble from multiple models.

    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained models
    voting : str
        'hard' or 'soft' voting
    weights : list, optional
        Weights for each model

    Returns:
    --------
    VotingClassifier
        Voting ensemble
    """
    estimators = [(name, model) for name, model in models_dict.items()]

    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1
    )

    return voting_clf


def get_stacking_ensemble(base_models, meta_model=None):
    """
    Create stacking ensemble.

    Parameters:
    -----------
    base_models : dict
        Dictionary of base models
    meta_model : estimator, optional
        Meta-learner model

    Returns:
    --------
    StackingClassifier
        Stacking ensemble
    """
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42, max_iter=1000)

    estimators = [(name, model) for name, model in base_models.items()]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        passthrough=False,
        n_jobs=-1
    )

    return stacking_clf


def get_model_by_name(name):
    """
    Get a specific model by name.

    Parameters:
    -----------
    name : str
        Model name

    Returns:
    --------
    estimator
        Model instance
    """
    models = {
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'dt': DecisionTreeClassifier(random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'nb': GaussianNB(),
        'svm': SVC(probability=True, random_state=42),
    }

    return models.get(name.lower())


def train_and_evaluate_models(X_train, y_train, X_val=None, y_val=None):
    """
    Train multiple models and return results.

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training labels
    X_val : DataFrame, optional
        Validation features
    y_val : Series, optional
        Validation labels

    Returns:
    --------
    dict
        Dictionary of trained models and their scores
    """
    from sklearn.model_selection import cross_val_score

    results = {}
    models = get_baseline_models()

    print("Training baseline models...")
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )

        # Train on full training set
        model.fit(X_train, y_train)

        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

        print(f"  {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


def get_feature_importance(model, feature_names):
    """
    Get feature importance from a model.

    Parameters:
    -----------
    model : estimator
        Trained model
    feature_names : list
        List of feature names

    Returns:
    --------
    DataFrame
        Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
    else:
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.zeros(len(feature_names))
        })

    return importance


if __name__ == '__main__':
    # Test model loading
    print("Testing model loading...")
    models = get_baseline_models()
    print(f"Loaded {len(models)} baseline models")
    models = get_advanced_models()
    print(f"Loaded {len(models)} advanced models")
