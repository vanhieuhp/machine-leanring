"""
Trainer Module
==============

Handles model training and validation.
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time


def train_with_validation(model, X_train, y_train, X_val, y_val, verbose=True):
    """
    Train model with validation set

    Returns:
        trained model, history dict
    """
    start_time = time.time()

    # Train
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    if verbose:
        print(f"Training time: {train_time:.2f}s")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Validation accuracy: {val_score:.4f}")

    history = {
        'train_time': train_time,
        'train_score': train_score,
        'val_score': val_score
    }

    return model, history


def cross_validate(model, X, y, cv=5, verbose=True):
    """
    Perform cross-validation

    Returns:
        cv_scores, mean_score, std_score
    """
    start_time = time.time()

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    cv_time = time.time() - start_time

    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    if verbose:
        print(f"Cross-validation completed in {cv_time:.2f}s")
        print(f"CV Scores: {cv_scores}")
        print(f"Mean: {mean_score:.4f} (+/- {std_score*2:.4f})")

    return cv_scores, mean_score, std_score


def train_with_early_stopping(model, X_train, y_train, X_val, y_val,
                             patience=10, verbose=True):
    """
    Train with early stopping for models that support it

    Note: This is a simplified version. For XGBoost/LightGBM,
    use their built-in early stopping.
    """
    best_val_score = 0
    best_model = None
    no_improvement_count = 0

    # Simple approach: train and check
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    if verbose:
        print(f"Train: {train_score:.4f}, Val: {val_score:.4f}")

    return model, {'train_score': train_score, 'val_score': val_score}


def train_ensemble(models, X_train, y_train, X_val=None, y_val=None, verbose=True):
    """
    Train multiple models and combine predictions

    Returns:
        trained models dict, results dict
    """
    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name}...")

        start_time = time.time()

        # Train
        model.fit(X_train, y_train)

        train_time = time.time() - start_time

        # Evaluate
        train_score = model.score(X_train, y_train)

        if X_val is not None:
            val_score = model.score(X_val, y_val)
        else:
            val_score = None

        results[name] = {
            'model': model,
            'train_time': train_time,
            'train_score': train_score,
            'val_score': val_score
        }

        if verbose:
            print(f"  Train: {train_score:.4f}, Val: {val_score:.4f if val_score else 'N/A'}")

    return models, results


def voting_ensemble(models, X_test, verbose=True):
    """
    Combine predictions using voting

    Returns:
        predictions
    """
    from collections import Counter

    all_predictions = []

    for name, result in models.items():
        model = result['model']
        predictions = model.predict(X_test)
        all_predictions.append(predictions)

    # Majority voting
    all_predictions = np.array(all_predictions)

    # For each sample, take majority vote
    final_predictions = []
    for i in range(len(X_test)):
        votes = all_predictions[:, i]
        counter = Counter(votes)
        final_predictions.append(counter.most_common(1)[0][0])

    return np.array(final_predictions)


def stacking_ensemble(models, X_train, y_train, X_test, meta_model=None, verbose=True):
    """
    Create stacking ensemble

    Returns:
        predictions
    """
    from sklearn.model_selection import cross_val_predict

    if meta_model is None:
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression(max_iter=1000)

    # Get out-of-fold predictions for stacking
    if verbose:
        print("Creating stacking features...")

    stacking_features_train = []
    stacking_features_test = []

    for name, result in models.items():
        model = result['model']

        # Get OOF predictions for training
        from sklearn.model_selection import cross_val_predict
        oof_pred = cross_val_predict(model, X_train, y_train, cv=5)

        if len(oof_pred.shape) > 1:
            oof_pred = oof_pred[:, 1]  # Get probability for class 1

        stacking_features_train.append(oof_pred)

        # Get predictions for test
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        if len(test_pred.shape) > 1:
            test_pred = test_pred[:, 1]

        stacking_features_test.append(test_pred)

    # Stack features
    X_stack_train = np.column_stack(stacking_features_train)
    X_stack_test = np.column_stack(stacking_features_test)

    # Train meta model
    if verbose:
        print("Training meta-learner...")

    meta_model.fit(X_stack_train, y_train)
    predictions = meta_model.predict(X_stack_test)

    return predictions


def save_model(model, filepath):
    """Save model to file"""
    import pickle

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load model from file"""
    import pickle

    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    return model


if __name__ == "__main__":
    # Test the trainer
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Test cross-validation
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    cv_scores, mean_score, std_score = cross_validate(model, X_train, y_train)

    print(f"\nFinal: {mean_score:.4f} (+/- {std_score*2:.4f})")
