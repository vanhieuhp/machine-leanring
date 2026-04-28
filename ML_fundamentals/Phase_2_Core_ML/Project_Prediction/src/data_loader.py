"""
Data Loader - Load and Prepare ML Datasets
=========================================

This module handles:
- Loading datasets (CSV, built-in)
- Data exploration
- Feature preprocessing
- Train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split

def load_iris_dataset():
    """
    Load the Iris dataset for classification.

    Returns:
        X (ndarray): Features
        y (ndarray): Target
        feature_names (list): Feature names
        target_names (list): Target class names
    """
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

def load_diabetes_dataset():
    """
    Load the Diabetes dataset for regression.

    Returns:
        X (ndarray): Features
        y (ndarray): Target
        feature_names (list): Feature names
    """
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target, diabetes.feature_names

def load_breast_cancer_dataset():
    """
    Load the Breast Cancer dataset for classification.

    Returns:
        X (ndarray): Features
        y (ndarray): Target
        feature_names (list): Feature names
        target_names (list): Target class names
    """
    cancer = load_breast_cancer()
    return cancer.data, cancer.target, cancer.feature_names, cancer.target_names

def load_csv_dataset(filepath, target_column=None):
    """
    Load dataset from CSV file.

    Args:
        filepath (str): Path to CSV file
        target_column (str): Name of target column (optional)

    Returns:
        df (DataFrame): Full dataframe
        X (ndarray): Features
        y (ndarray): Target (if target_column provided)
    """
    df = pd.read_csv(filepath)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return df, X.values, y.values

    return df, None, None

def explore_data(X, y, feature_names=None, target_names=None):
    """
    Perform initial exploration of the dataset.

    Args:
        X: Features array
        y: Target array
        feature_names: List of feature names
        target_names: List of target class names
    """
    print("=" * 70)
    print("DATA EXPLORATION")
    print("=" * 70)

    print(f"\nDataset Shape:")
    print(f"  Features: {X.shape}")
    print(f"  Target: {y.shape}")

    print(f"\nFeature Names:")
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            print(f"  {i}: {name}")
    else:
        for i in range(X.shape[1]):
            print(f"  {i}: Feature {i}")

    print(f"\nTarget Distribution:")
    if target_names is not None:
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {target_names[u]}: {c} ({c/len(y)*100:.1f}%)")
    else:
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} ({c/len(y)*100:.1f}%)")

    print(f"\nFeature Statistics:")
    df = pd.DataFrame(X, columns=feature_names)
    print(df.describe())

def prepare_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def check_missing_values(X, feature_names=None):
    """
    Check for missing values in the dataset.

    Args:
        X: Features array
        feature_names: List of feature names
    """
    print("=" * 70)
    print("MISSING VALUES CHECK")
    print("=" * 70)

    df = pd.DataFrame(X, columns=feature_names)
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("\nNo missing values found!")
    else:
        print("\nMissing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count}")

def get_feature_types(X, feature_names=None):
    """
    Identify feature types (numeric vs categorical).

    Args:
        X: Features array
        feature_names: List of feature names
    """
    print("=" * 70)
    print("FEATURE TYPES")
    print("=" * 70)

    df = pd.DataFrame(X, columns=feature_names)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"\nNumeric Features ({len(numeric_cols)}):")
    for col in numeric_cols[:10]:
        print(f"  - {col}")
    if len(numeric_cols) > 10:
        print(f"  ... and {len(numeric_cols) - 10} more")

    print(f"\nCategorical Features ({len(categorical_cols)}):")
    if categorical_cols:
        for col in categorical_cols:
            print(f"  - {col}: {df[col].nunique()} unique values")
    else:
        print("  None")

if __name__ == "__main__":
    # Demo with Iris dataset
    print("Loading Iris dataset...")
    X, y, feature_names, target_names = load_iris_dataset()

    explore_data(X, y, feature_names, target_names)
    check_missing_values(X, feature_names)

    print("\n\nSplitting data...")
    X_train, X_test, y_train, y_test = prepare_train_test(X, y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
