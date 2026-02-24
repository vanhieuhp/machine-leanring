"""
Exploratory Analysis - Statistical Analysis of Iris Dataset
============================================================

This module handles:
- Statistical analysis
- Correlation analysis
- Outlier detection
- Data quality checks
"""

import pandas as pd
import numpy as np
from scipy import stats

def analyze_features(df):
    """
    Analyze each feature in detail.

    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col not in ['target', 'target_name']]

    for col in feature_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Median: {df[col].median():.3f}")
        print(f"  Std Dev: {df[col].std():.3f}")
        print(f"  Min: {df[col].min():.3f}")
        print(f"  Max: {df[col].max():.3f}")
        print(f"  Skewness: {stats.skew(df[col]):.3f}")
        print(f"  Kurtosis: {stats.kurtosis(df[col]):.3f}")

def analyze_by_class(df):
    """
    Analyze features by target class.

    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("\n" + "=" * 70)
    print("ANALYSIS BY CLASS")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col not in ['target', 'target_name']]

    for target in df['target_name'].unique():
        print(f"\n{target.upper()}:")
        subset = df[df['target_name'] == target]

        for col in feature_cols:
            print(f"  {col}: {subset[col].mean():.3f} ± {subset[col].std():.3f}")

def analyze_correlations(df):
    """
    Analyze correlations between features.

    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col not in ['target', 'target_name']]
    corr_matrix = df[feature_cols].corr()

    print("\nCorrelation Matrix:")
    print(corr_matrix)

    print("\nHighest Correlations:")
    # Get upper triangle of correlation matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values('Correlation', ascending=False, key=abs)

    for idx, row in corr_df.head(5).iterrows():
        print(f"  {row['Feature 1']} <-> {row['Feature 2']}: {row['Correlation']:.3f}")

def detect_outliers(df):
    """
    Detect outliers using IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION (IQR Method)")
    print("=" * 70)

    feature_cols = [col for col in df.columns if col not in ['target', 'target_name']]

    outlier_count = 0

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        if len(outliers) > 0:
            print(f"\n{col}:")
            print(f"  Bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
            print(f"  Outliers found: {len(outliers)}")
            outlier_count += len(outliers)

    if outlier_count == 0:
        print("\nNo outliers detected!")

def check_data_quality(df):
    """
    Check overall data quality.

    Args:
        df (pd.DataFrame): Input DataFrame
    """
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print(f"\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('iris.csv')

    # Perform analysis
    analyze_features(df)
    analyze_by_class(df)
    analyze_correlations(df)
    detect_outliers(df)
    check_data_quality(df)
