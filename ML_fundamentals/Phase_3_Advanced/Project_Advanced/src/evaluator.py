"""
Evaluator Module
===============

Handles model evaluation and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import seaborn as sns


def evaluate_classification(y_true, y_pred, verbose=True):
    """Evaluate classification model"""

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

    if verbose:
        print("Classification Metrics:")
        print("-" * 40)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

    return metrics


def evaluate_regression(y_true, y_pred, verbose=True):
    """Evaluate regression model"""

    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    if verbose:
        print("Regression Metrics:")
        print("-" * 40)
        print(f"MSE:   {metrics['mse']:.4f}")
        print(f"RMSE:  {metrics['rmse']:.4f}")
        print(f"MAE:   {metrics['mae']:.4f}")
        print(f"R²:    {metrics['r2']:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
    """Plot confusion matrix"""

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, class_names=None):
    """Plot ROC curve"""

    plt.figure(figsize=(8, 6))

    if len(np.unique(y_true)) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Multi-class
        for i, name in enumerate(class_names or range(y_proba.shape[1])):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=None):
    """Plot feature importance"""

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    if top_n:
        indices = indices[:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)),
             importances[indices],
             color='steelblue')
    plt.yticks(range(len(indices)),
               [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(train_sizes, train_scores, val_scores):
    """Plot learning curve"""

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='green', label='Validation score')
    plt.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.1, color='green')

    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(model_names, scores, metric_name='Accuracy'):
    """Plot model comparison"""

    plt.figure(figsize=(12, 6))
    plt.barh(model_names, scores, color='steelblue')
    plt.xlabel(metric_name)
    plt.title(f'Model Comparison - {metric_name}')
    plt.xlim(0, 1)

    # Add value labels
    for i, (name, score) in enumerate(zip(model_names, scores)):
        plt.text(score + 0.01, i, f'{score:.4f}', va='center')

    plt.tight_layout()
    plt.show()


def generate_classification_report(y_true, y_pred, class_names=None):
    """Generate classification report"""

    report = classification_report(y_true, y_pred,
                                 target_names=class_names,
                                 output_dict=True)

    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    return report


def plot_predictions(y_true, y_pred, title='Predictions vs Actual'):
    """Plot predictions vs actual values"""

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred):
    """Plot residuals for regression"""

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predictions
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predictions')

    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test evaluation
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Evaluate
    metrics = evaluate_classification(y_test, y_pred)
    print(f"\nFinal Accuracy: {metrics['accuracy']:.4f}")
