"""
Data Loader - Load and Explore Iris Dataset
=============================================

This module handles:
- Loading the Iris dataset
- Initial exploration
- Data structure overview
"""

import pandas as pd
import numpy as np
from sklearn import datasets

def load_iris_data():
    """
    Load the Iris dataset from scikit-learn.

    Returns:
        df (pd.DataFrame): DataFrame with features and target
    """
    # Load dataset
    iris = datasets.load_iris()

    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map({
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    })

    return df

def explore_data(df):
    """
    Perform initial exploration of the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n" + "=" * 70)
    print("FIRST FEW ROWS")
    print("=" * 70)
    print(df.head(10))

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)
    print(df.dtypes)

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)
    print(df.isnull().sum())

    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)
    print(df.describe())

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)
    print(df['target_name'].value_counts())

if __name__ == "__main__":
    # Load data
    df = load_iris_data()

    # Explore
    explore_data(df)

    # Save for later use
    df.to_csv('iris.csv', index=False)
    print("\n\nData saved to iris.csv")
