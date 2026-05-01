"""
Model Evaluation - Evaluate and Visualize Model Performance
=========================================================

This module handles:
- Comprehensive model evaluation
- Visualization of results
- Performance comparison
- Error analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc,
                           roc_auc_score, mean_squared_error, mean_absolute_error,
                           r2_score, silhouette_score, silhouette_samples)
from sklearn.model_selection import cross_val_score
import seaborn as sns

class ModelEvaluator:
    """Class to evaluate and visualize model performance."""

    def __init__(self, X_train, y_train, X_test, y_test, task_type='classification'):
        """
        Initialize ModelEvaluator.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            task_type: 'classification', 'regression', or 'clustering'
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type

    def evaluate_classification(self, model, model_name='Model', plot=True):
        """
        Comprehensive classification evaluation.

        Args:
            model: Trained model
            model_name: Name for display
            plot: Whether to create plots

        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }

        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_proba)

        # Print metrics
        print("=" * 70)
        print(f"CLASSIFICATION EVALUATION: {model_name}")
        print("=" * 70)

        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # Plots
        if plot:
            self._plot_classification_results(y_pred, y_proba)

        return metrics

    def _plot_classification_results(self, y_pred, y_proba):
        """Create classification evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        ax = axes[0, 0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # 2. ROC Curve (if probabilities available)
        ax = axes[0, 1]
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No probabilities available',
                   ha='center', va='center', fontsize=14)
            ax.set_title('ROC Curve')

        # 3. Feature Importance (if available)
        ax = axes[1, 0]
        if hasattr(self.X_train, 'columns') and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            features = self.X_train.columns
            ax.barh(features, importances, color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Feature importance not available',
                   ha='center', va='center', fontsize=14)
            ax.set_title('Feature Importance')

        # 4. Metrics Summary
        ax = axes[1, 1]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metrics_values = [self.metrics['accuracy'], self.metrics['precision'],
                         self.metrics['recall'], self.metrics['f1']]
        colors = ['blue', 'green', 'orange', 'red']
        ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        for i, v in enumerate(metrics_values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    def evaluate_regression(self, model, model_name='Model', plot=True):
        """
        Comprehensive regression evaluation.

        Args:
            model: Trained model
            model_name: Name for display
            plot: Whether to create plots

        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(self.X_test)

        # Metrics
        metrics = {
            'mse': mean_squared_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'r2': r2_score(self.y_test, y_pred)
        }

        # Print metrics
        print("=" * 70)
        print(f"REGRESSION EVALUATION: {model_name}")
        print("=" * 70)

        print(f"\nMSE:  {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"R²:   {metrics['r2']:.4f}")

        # Plots
        if plot:
            self._plot_regression_results(y_pred)

        return metrics

    def _plot_regression_results(self, y_pred):
        """Create regression evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(self.y_test, y_pred, alpha=0.5)
        ax.plot([self.y_test.min(), self.y_test.max()],
               [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        ax.grid(True, alpha=0.3)

        # 2. Residuals
        ax = axes[0, 1]
        residuals = self.y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)

        # 3. Residuals Distribution
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residuals Distribution')
        ax.grid(True, alpha=0.3)

        # 4. Metrics Summary
        ax = axes[1, 1]
        metrics_names = ['RMSE', 'MAE', 'R²']
        metrics_values = [self.metrics['rmse'], self.metrics['mae'], self.metrics['r2']]
        colors = ['blue', 'green', 'orange']
        ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        for i, v in enumerate(metrics_values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    def evaluate_clustering(self, model, X, model_name='Model', plot=True):
        """
        Comprehensive clustering evaluation.

        Args:
            model: Trained clustering model
            X: Features (original, not scaled if scaled was used)
            model_name: Name for display
            plot: Whether to create plots

        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        labels = model.predict(X) if hasattr(model, 'predict') else model.labels_

        # Metrics
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'n_clusters': len(np.unique(labels))
        }

        # Print metrics
        print("=" * 70)
        print(f"CLUSTERING EVALUATION: {model_name}")
        print("=" * 70)

        print(f"\nNumber of Clusters: {metrics['n_clusters']}")
        print(f"Silhouette Score: {metrics['silhouette']:.4f}")

        # Plots
        if plot:
            self._plot_clustering_results(X, labels)

        return metrics

    def _plot_clustering_results(self, X, labels):
        """Create clustering evaluation plots."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Clusters
        ax = axes[0]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        if hasattr(self.model, 'cluster_centers_'):
            centers = self.model.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200,
                      edgecolors='black', linewidths=2, label='Centroids')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Cluster Assignments')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # 2. Silhouette Analysis
        ax = axes[1]
        silhouette_vals = silhouette_samples(X, labels)
        y_lower = 10
        for i in range(len(np.unique(labels))):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, cluster_silhouette_vals,
                           alpha=0.7, label=f'Cluster {i}')
            y_lower = y_upper + 10

        avg_silhouette = silhouette_score(X, labels)
        ax.axvline(x=avg_silhouette, color='red', linestyle='--',
                  label=f'Avg: {avg_silhouette:.4f}')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.set_title('Silhouette Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_models(self, results_dict):
        """
        Compare multiple models.

        Args:
            results_dict: Dictionary of {model_name: metrics_dict}
        """
        print("=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        if self.task_type == 'classification':
            print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
                "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"))
            print("-" * 75)

            for name, metrics in results_dict.items():
                roc_auc = metrics.get('roc_auc', 0)
                print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    name,
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1'],
                    roc_auc
                ))

        elif self.task_type == 'regression':
            print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
                "Model", "MSE", "RMSE", "MAE", "R²"))
            print("-" * 65)

            for name, metrics in results_dict.items():
                print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    name,
                    metrics['mse'],
                    metrics['rmse'],
                    metrics['mae'],
                    metrics['r2']
                ))

    def cross_validate(self, model, cv=5):
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            cv: Number of folds

        Returns:
            dict: CV results
        """
        if self.task_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'

        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)

        print("=" * 70)
        print(f"CROSS-VALIDATION RESULTS (cv={cv})")
        print("=" * 70)
        print(f"\nFold Scores: {scores}")
        print(f"Mean: {scores.mean():.4f}")
        print(f"Std: {scores.std():.4f}")

        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }


def create_evaluation_report(results_dict, task_type='classification'):
    """
    Create a comprehensive evaluation report.

    Args:
        results_dict: Dictionary of model results
        task_type: Type of task

    Returns:
        DataFrame: Summary table
    """
    df = pd.DataFrame(results_dict).T
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(df)


if __name__ == "__main__":
    # Demo with Iris dataset
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("MODEL EVALUATION DEMO")
    print("=" * 70)

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    evaluator = ModelEvaluator(X_train, y_train, X_test, y_test, 'classification')
    metrics = evaluator.evaluate_classification(model, 'Logistic Regression')
