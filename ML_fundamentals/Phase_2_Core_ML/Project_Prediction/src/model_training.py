"""
Model Training - Train Multiple ML Models
=======================================

This module handles:
- Training multiple ML models
- Hyperparameter tuning
- Model comparison
- Cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, r2_score, confusion_matrix,
                           roc_curve, auc, silhouette_score)

class ModelTrainer:
    """Class to handle model training and comparison."""

    def __init__(self, X_train, y_train, X_test, y_test, task_type='classification'):
        """
        Initialize ModelTrainer.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            task_type: 'classification' or 'regression' or 'clustering'
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        """Add a model to compare."""
        self.models[name] = model

    def train_all(self):
        """Train all models."""
        print("=" * 70)
        print("TRAINING MODELS")
        print("=" * 70)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(self.X_train, self.y_train)

            # Get predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            # Calculate metrics based on task type
            if self.task_type == 'classification':
                results = self._classification_metrics(y_pred_train, y_pred_test)
            elif self.task_type == 'regression':
                results = self._regression_metrics(y_pred_train, y_pred_test)
            else:
                results = {}

            self.results[name] = {
                'model': model,
                'predictions': y_pred_test,
                **results
            }

            print(f"  Training complete!")

    def _classification_metrics(self, y_pred_train, y_pred_test):
        """Calculate classification metrics."""
        return {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
            'precision': precision_score(self.y_test, y_pred_test, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_test, average='weighted'),
            'f1': f1_score(self.y_test, y_pred_test, average='weighted')
        }

    def _regression_metrics(self, y_pred_train, y_pred_test):
        """Calculate regression metrics."""
        return {
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test)
        }

    def compare_models(self):
        """Print comparison of all models."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        if self.task_type == 'classification':
            print("\n{:<25} {:>12} {:>12} {:>10} {:>10} {:>10}".format(
                "Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1"))
            print("-" * 80)

            for name, result in self.results.items():
                print("{:<25} {:>12.4f} {:>12.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    name,
                    result['train_accuracy'],
                    result['test_accuracy'],
                    result['precision'],
                    result['recall'],
                    result['f1']
                ))

        elif self.task_type == 'regression':
            print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
                "Model", "Train RMSE", "Test RMSE", "Train R²", "Test R²"))
            print("-" * 70)

            for name, result in self.results.items():
                print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                    name,
                    result['train_rmse'],
                    result['test_rmse'],
                    result['train_r2'],
                    result['test_r2']
                ))

    def get_best_model(self, metric='test_accuracy'):
        """Get the best model based on a metric."""
        if self.task_type == 'classification':
            if metric == 'test_accuracy':
                best_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['test_accuracy'])
            elif metric == 'f1':
                best_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['f1'])
            else:
                best_name = max(self.results.keys(),
                              key=lambda x: self.results[x][metric])
        else:
            if metric == 'test_r2':
                best_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['test_r2'])
            elif metric == 'test_rmse':
                best_name = min(self.results.keys(),
                              key=lambda x: self.results[x]['test_rmse'])
            else:
                best_name = max(self.results.keys(),
                              key=lambda x: self.results[x][metric])

        print(f"\nBest Model: {best_name}")
        return self.results[best_name]['model']

    def cross_validate(self, cv=5):
        """Perform cross-validation on all models."""
        print("\n" + "=" * 70)
        print(f"CROSS-VALIDATION (cv={cv})")
        print("=" * 70)

        for name, model in self.models.items():
            if self.task_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'r2'

            scores = cross_val_score(model, self.X_train, self.y_train,
                                   cv=cv, scoring=scoring)

            print(f"\n{name}:")
            print(f"  CV Scores: {scores}")
            print(f"  Mean: {scores.mean():.4f}")
            print(f"  Std: {scores.std():.4f}")

    def hyperparameter_tuning(self, param_grid, cv=5):
        """Perform hyperparameter tuning on the best model."""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING")
        print("=" * 70)

        # Use first model for tuning
        name = list(self.models.keys())[0]
        model = self.models[name]

        print(f"\nTuning {name}...")
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy' if
                                  self.task_type == 'classification' else 'r2',
                                  verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_


def create_classification_models():
    """Create dictionary of classification models."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }


def create_regression_models():
    """Create dictionary of regression models."""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }


def train_classification_pipeline(X, y, test_size=0.2, random_state=42):
    """
    Complete classification training pipeline.

    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        trainer: ModelTrainer object with trained models
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create trainer
    trainer = ModelTrainer(X_train, y_train, X_test, y_test, 'classification')

    # Add models
    models = create_classification_models()
    for name, model in models.items():
        trainer.add_model(name, model)

    # Train all
    trainer.train_all()

    # Compare
    trainer.compare_models()

    # Cross-validation
    trainer.cross_validate(cv=5)

    return trainer


def train_regression_pipeline(X, y, test_size=0.2, random_state=42):
    """
    Complete regression training pipeline.

    Args:
        X: Features
        y: Target
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        trainer: ModelTrainer object with trained models
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create trainer
    trainer = ModelTrainer(X_train, y_train, X_test, y_test, 'regression')

    # Add models
    models = create_regression_models()
    for name, model in models.items():
        trainer.add_model(name, model)

    # Train all
    trainer.train_all()

    # Compare
    trainer.compare_models()

    # Cross-validation
    trainer.cross_validate(cv=5)

    return trainer


if __name__ == "__main__":
    # Demo with Iris dataset
    from sklearn.datasets import load_iris

    print("=" * 70)
    print("CLASSIFICATION PIPELINE DEMO")
    print("=" * 70)

    iris = load_iris()
    X, y = iris.data, iris.target

    trainer = train_classification_pipeline(X, y)

    # Get best model
    best = trainer.get_best_model(metric='test_accuracy')
    print(f"\nBest model type: {type(best).__name__}")
