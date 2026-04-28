"""
Data Loader Module
=================

Handles data loading and preprocessing for the advanced ML project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_dataset(filepath):
    """Load dataset from CSV file"""
    df = pd.read_csv(filepath)
    return df


def load_titanic():
    """Load Titanic dataset for demonstration"""
    # Try to load from sklearn or use a sample
    try:
        # Try OpenML
        from sklearn.datasets import fetch_openml
        titanic = fetch_openml('titanic', version=1, as_frame=True)
        df = titanic.frame
    except:
        # Create sample data
        print("Using sample Titanic-like data")
        df = create_sample_titanic()

    return df


def create_sample_titanic():
    """Create sample Titanic-like dataset"""
    np.random.seed(42)
    n = 891

    data = {
        'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n),
        'age': np.random.normal(30, 15, n).clip(0, 80),
        'sibsp': np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.2, 0.1, 0.06, 0.04]),
        'parch': np.random.choice([0, 1, 2, 3], n, p=[0.7, 0.15, 0.1, 0.05]),
        'fare': np.random.exponential(30, n),
        'embarked': np.random.choice(['C', 'Q', 'S'], n, p=[0.2, 0.1, 0.7]),
        'survived': np.random.choice([0, 1], n, p=[0.6, 0.4])
    }

    df = pd.DataFrame(data)
    return df


def preprocess_data(df, target_column='survived'):
    """Preprocess the dataset"""

    # Make a copy
    df = df.copy()

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Fill numeric missing values with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing values with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def encode_categorical(df, columns):
    """Encode categorical columns"""

    df = df.copy()
    label_encoders = {}

    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def get_features_target(df, target_column='survived'):
    """Extract features and target from dataframe"""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def preprocess_pipeline(df, target_column='survived', categorical_columns=None,
                       test_size=0.2, scale=True):
    """
    Complete preprocessing pipeline

    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoders
    """

    # Preprocess
    df = preprocess_data(df, target_column)

    # Encode categorical
    if categorical_columns:
        df, label_encoders = encode_categorical(df, categorical_columns)
    else:
        # Auto-detect categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        df, label_encoders = encode_categorical(df, categorical_columns)

    # Get features and target
    X, y = get_features_target(df, target_column)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    # Scale
    if scale:
        X_train, X_test, scaler = scale_features(X_train, X_test)
    else:
        scaler = None

    return X_train, X_test, y_train, y_test, scaler, label_encoders


if __name__ == "__main__":
    # Test the data loader
    df = load_titanic()
    print(f"Loaded dataset: {df.shape}")
    print(df.head())
    print(f"\nMissing values:\n{df.isnull().sum()}")
