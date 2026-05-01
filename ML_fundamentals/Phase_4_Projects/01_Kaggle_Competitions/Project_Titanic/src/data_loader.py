"""
Data Loader Module
=================

Handles loading and initial processing of Titanic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_dir='data'):
    """
    Load Titanic training and test data.

    Parameters:
    -----------
    data_dir : str
        Path to directory containing train.csv and test.csv

    Returns:
    --------
    train_df, test_df : DataFrame
        Training and test datasets
    """
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    return train_df, test_df


def get_data_info(df):
    """
    Display basic information about the dataset.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    """
    print("\n" + "=" * 50)
    print("DATA INFO")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst few rows:\n{df.head()}")


def combine_train_test(train_df, test_df):
    """
    Combine train and test data for consistent feature engineering.

    Parameters:
    -----------
    train_df : DataFrame
        Training data
    test_df : DataFrame
        Test data

    Returns:
    --------
    combined_df : DataFrame
        Combined dataframe
    """
    # Mark train and test
    train_df['is_train'] = 1
    test_df['is_train'] = 0

    # Combine
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    print(f"Combined shape: {combined.shape}")

    return combined


def save_submission(predictions, passenger_ids, output_path='submission.csv'):
    """
    Save predictions in Kaggle submission format.

    Parameters:
    -----------
    predictions : array-like
        Model predictions
    passenger_ids : array-like
        Passenger IDs
    output_path : str
        Output file path
    """
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions.astype(int)
    })

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())


if __name__ == '__main__':
    # Test data loading
    print("Testing data loader...")
    # In practice, this would load actual data
    print("This is a module - import it to use")
