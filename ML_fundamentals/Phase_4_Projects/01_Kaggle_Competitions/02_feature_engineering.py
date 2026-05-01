"""
Kaggle Competitions - Part 2: Feature Engineering
===================================================

This module covers:
- Handling missing values
- Encoding categorical variables
- Feature transformation
- Creating new features
- Feature scaling
- Feature selection

Based on: Titanic Competition
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. HANDLING MISSING VALUES
# ============================================================================

print("=" * 70)
print("1. HANDLING MISSING VALUES")
print("=" * 70)

# Create sample data with missing values
np.random.seed(42)
n = 100

data = {
    'Age': np.random.uniform(18, 70, n),
    'Salary': np.random.uniform(30000, 150000, n),
    'Department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], n),
    'Score': np.random.choice([1, 2, 3, 4, 5], n),
    'Experience': np.random.randint(0, 30, n)
}

df = pd.DataFrame(data)

# Introduce missing values
df.loc[np.random.choice(n, 10, replace=False), 'Age'] = np.nan
df.loc[np.random.choice(n, 15, replace=False), 'Salary'] = np.nan
df.loc[np.random.choice(n, 5, replace=False), 'Department'] = np.nan
df.loc[np.random.choice(n, 8, replace=False), 'Experience'] = np.nan

print("Original data with missing values:")
print(df.head(10))
print(f"\nMissing values:\n{df.isnull().sum()}")

# -------------------------------------------------------------------------
# 1.1 Simple Imputation (Mean/Median/Mode)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.1 Mean/Median/Mode Imputation")
print("-" * 50)

# Mean imputation (for numerical)
df_mean = df.copy()
df_mean['Age'].fillna(df_mean['Age'].mean(), inplace=True)
df_mean['Salary'].fillna(df_mean['Salary'].mean(), inplace=True)

# Median imputation (better for skewed data)
df_median = df.copy()
df_median['Age'].fillna(df_median['Age'].median(), inplace=True)
df_median['Salary'].fillna(df_median['Salary'].median(), inplace=True)

# Mode imputation (for categorical)
df_mode = df.copy()
df_mode['Department'].fillna(df_mode['Department'].mode()[0], inplace=True)

print("Mean imputation:")
print(df_mean[['Age', 'Salary']].head())

print("\nMedian imputation:")
print(df_median[['Age', 'Salary']].head())

print("\nMode imputation:")
print(df_mode['Department'].value_counts())

# -------------------------------------------------------------------------
# 1.2 Forward/Backward Fill (for time series / ordered data)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.2 Forward/Backward Fill")
print("-" * 50)

df_ordered = pd.DataFrame({
    'Value': [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9]
})

df_ffill = df_ordered.fillna(method='ffill')
df_bfill = df_ordered.fillna(method='bfill')

print("Original:", df_ordered['Value'].values)
print("Forward fill:", df_ffill['Value'].values)
print("Backward fill:", df_bfill['Value'].values)

# -------------------------------------------------------------------------
# 1.3 KNN Imputation
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.3 KNN Imputation")
print("-" * 50)

# Create sample data for KNN imputation
sample_data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, np.nan, 55, 60],
    'Salary': [50000, 55000, np.nan, 65000, 70000, 75000, np.nan, 90000],
    'Score': [3, 4, 5, np.nan, 4, 3, np.nan, 5]
})

# Use sklearn's KNNImputer
imputer = KNNImputer(n_neighbors=3)
imputed_data = imputer.fit_transform(sample_data)
df_imputed = pd.DataFrame(imputed_data, columns=sample_data.columns)

print("Original with NaN:")
print(sample_data)
print("\nAfter KNN Imputation:")
print(df_imputed)

# ============================================================================
# 2. ENCODING CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "=" * 70)
print("2. ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# Sample data for encoding
df_cat = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red'],
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large', 'Medium'],
    'Category': ['A', 'B', 'C', 'A', 'B', 'C', 'A'],
    'Target': [1, 0, 1, 0, 1, 0, 1]
})

# -------------------------------------------------------------------------
# 2.1 Label Encoding (Ordinal/Nominal with order)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.1 Label Encoding")
print("-" * 50)

# Size has natural order: Small < Medium < Large
size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
df_cat['Size_encoded'] = df_cat['Size'].map(size_mapping)

print("Label encoding (Size):")
print(df_cat[['Size', 'Size_encoded']])

# Using sklearn
le = LabelEncoder()
df_cat['Color_encoded'] = le.fit_transform(df_cat['Color'])
print("\nLabel encoding (Color):")
print(df_cat[['Color', 'Color_encoded']])

# -------------------------------------------------------------------------
# 2.2 One-Hot Encoding (Nominal - no natural order)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.2 One-Hot Encoding")
print("-" * 50)

# Using pandas
df_onehot = pd.get_dummies(df_cat, columns=['Color'], prefix=['Color'])
print("One-Hot Encoding (pandas get_dummies):")
print(df_onehot[['Color_Red', 'Color_Blue', 'Color_Green']])

# Using sklearn
ohe = OneHotEncoder(sparse_output=False, drop='first')
color_encoded = ohe.fit_transform(df_cat[['Color']])
color_cols = ohe.get_feature_names_out(['Color'])
df_ohe_sklearn = pd.DataFrame(color_encoded, columns=color_cols)
print("\nOne-Hot Encoding (sklearn):")
print(df_ohe_sklearn)

# -------------------------------------------------------------------------
# 2.3 Target Encoding (For high cardinality)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2.3 Target Encoding")
print("-" * 50)

# Create sample with high cardinality
df_high_card = pd.DataFrame({
    'City': ['NY', 'LA', 'NY', 'Chicago', 'LA', 'NY', 'Chicago', 'LA'],
    'Target': [1, 0, 1, 0, 1, 0, 1, 0]
})

# Calculate mean target for each city
city_means = df_high_card.groupby('City')['Target'].mean()
global_mean = df_high_card['Target'].mean()

df_high_card['City_target_encoded'] = df_high_card['City'].map(city_means)

print("Target Encoding:")
print(df_high_card[['City', 'Target', 'City_target_encoded']])
print(f"\nCity means: {city_means.to_dict()}")
print(f"Global mean: {global_mean}")

# ============================================================================
# 3. FEATURE TRANSFORMATION
# ============================================================================

print("\n" + "=" * 70)
print("3. FEATURE TRANSFORMATION")
print("=" * 70)

# Sample skewed data
df_skewed = pd.DataFrame({
    'Income': [20000, 25000, 30000, 35000, 40000, 50000, 75000, 100000, 150000, 500000],
    'Age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
})

print("Original data:")
print(df_skewed)
print(f"\nIncome skewness: {df_skewed['Income'].skew():.2f}")

# -------------------------------------------------------------------------
# 3.1 Log Transformation
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.1 Log Transformation")
print("-" * 50)

# Log transform (handle 0/negative values)
df_skewed['Income_log'] = np.log1p(df_skewed['Income'])  # log(1+x) to handle zeros

print("After log transformation:")
print(df_skewed[['Income', 'Income_log']])
print(f"Log Income skewness: {df_skewed['Income_log'].skew():.2f}")

# -------------------------------------------------------------------------
# 3.2 Square Root Transformation
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.2 Square Root Transformation")
print("-" * 50)

df_skewed['Income_sqrt'] = np.sqrt(df_skewed['Income'])

print("After sqrt transformation:")
print(df_skewed[['Income', 'Income_sqrt']])
print(f"Sqrt Income skewness: {df_skewed['Income_sqrt'].skew():.2f}")

# -------------------------------------------------------------------------
# 3.3 Box-Cox Transformation
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3.3 Box-Cox Transformation")
print("-" * 50)

from scipy import stats

# Box-Cox requires positive values
df_skewed['Income_boxcox'], lambda_val = stats.boxcox(df_skewed['Income'])

print(f"Box-Cox lambda: {lambda_val:.4f}")
print("After Box-Cox transformation:")
print(df_skewed[['Income', 'Income_boxcox']])
print(f"Box-Cox Income skewness: {df_skewed['Income_boxcox'].skew():.2f}")

# ============================================================================
# 4. CREATING NEW FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("4. CREATING NEW FEATURES (Feature Engineering)")
print("=" * 70)

# Titanic-like dataset
titanic = pd.DataFrame({
    'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina'],
    'Sex': ['male', 'female', 'female'],
    'Age': [22.0, 38.0, 26.0],
    'SibSp': [1, 1, 0],
    'Parch': [0, 0, 0],
    'Fare': [7.25, 71.2833, 7.925],
    'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
    'Cabin': [np.nan, 'C85', np.nan],
    'Embarked': ['S', 'C', 'S']
})

# -------------------------------------------------------------------------
# 4.1 Extract Titles from Names
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.1 Extract Titles from Names")
print("-" * 50)

def extract_title(name):
    """Extract title from name."""
    if 'Mr.' in name:
        return 'Mr'
    elif 'Mrs.' in name:
        return 'Mrs'
    elif 'Miss' in name:
        return 'Miss'
    elif 'Master' in name:
        return 'Master'
    else:
        return 'Other'

titanic['Title'] = titanic['Name'].apply(extract_title)
print("Extracted titles:")
print(titanic[['Name', 'Title']])

# -------------------------------------------------------------------------
# 4.2 Family Size Feature
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.2 Family Size Feature")
print("-" * 50)

titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1  # +1 for self

def family_category(size):
    if size == 1:
        return 'Alone'
    elif size <= 4:
        return 'Small'
    else:
        return 'Large'

titanic['FamilyCategory'] = titanic['FamilySize'].apply(family_category)

print("Family features:")
print(titanic[['SibSp', 'Parch', 'FamilySize', 'FamilyCategory']])

# -------------------------------------------------------------------------
# 4.3 Cabin Deck (from Cabin number)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.3 Cabin Deck Feature")
print("-" * 50)

titanic['HasCabin'] = titanic['Cabin'].notna().astype(int)
titanic['Deck'] = titanic['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')  # U = Unknown

print("Cabin features:")
print(titanic[['Cabin', 'HasCabin', 'Deck']])

# -------------------------------------------------------------------------
# 4.4 Binning/Discretization
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.4 Binning Continuous Variables")
print("-" * 50)

# Age binning
titanic['AgeBin'] = pd.cut(titanic['Age'],
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])

# Fare binning
titanic['FareBin'] = pd.qcut(titanic['Fare'].fillna(titanic['Fare'].median()),
                              q=3,
                              labels=['Low', 'Medium', 'High'])

print("Binned features:")
print(titanic[['Age', 'AgeBin', 'Fare', 'FareBin']])

# -------------------------------------------------------------------------
# 4.5 Feature Interactions
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4.5 Feature Interactions")
print("-" * 50)

# Create interaction features
titanic['Age*Class'] = titanic['Age'] * titanic.get('Pclass', pd.Series([3, 1, 3]))
titanic['FarePerPerson'] = titanic['Fare'] / titanic['FamilySize']

print("Interaction features:")
print(titanic[['Age', 'Fare', 'FamilySize', 'Age*Class', 'FarePerPerson']])

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================

print("\n" + "=" * 70)
print("5. FEATURE SCALING")
print("=" * 70)

sample_data = pd.DataFrame({
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90],
    'Age': [20, 25, 30, 35, 40]
})

# -------------------------------------------------------------------------
# 5.1 Standardization (Z-score)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.1 StandardScaler (Z-score normalization)")
print("-" * 50)

scaler = StandardScaler()
standardized = scaler.fit_transform(sample_data)
df_standardized = pd.DataFrame(standardized, columns=sample_data.columns)

print("Original:")
print(sample_data)
print("\nStandardized (mean=0, std=1):")
print(df_standardized)
print(f"\nMean after scaling: {df_standardized.mean().values}")
print(f"Std after scaling: {df_standardized.std().values}")

# -------------------------------------------------------------------------
# 5.2 Min-Max Scaling
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.2 MinMaxScaler (0-1 range)")
print("-" * 50)

minmax_scaler = MinMaxScaler()
minmax = minmax_scaler.fit_transform(sample_data)
df_minmax = pd.DataFrame(minmax, columns=sample_data.columns)

print("Min-Max scaled (0-1):")
print(df_minmax)
print(f"\nMin after scaling: {df_minmax.min().values}")
print(f"Max after scaling: {df_minmax.max().values}")

# -------------------------------------------------------------------------
# 5.3 Robust Scaling (using IQR)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.3 RobustScaler (using IQR)")
print("-" * 50)

from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
robust = robust_scaler.fit_transform(sample_data)
df_robust = pd.DataFrame(robust, columns=sample_data.columns)

print("Robust scaled (uses median and IQR):")
print(df_robust)

# ============================================================================
# 6. FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 70)
print("6. FEATURE SELECTION")
print("=" * 70)

# Create sample data with features
np.random.seed(42)
X = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Feature3': np.random.randn(100),
    'Feature4': np.random.randn(100),  # Unimportant
    'Feature5': np.random.randn(100),  # Unimportant
})

# Add some correlation
X['Feature2'] = X['Feature1'] * 0.5 + np.random.randn(100) * 0.1
X['Target'] = X['Feature1'] * 2 + X['Feature2'] * 1 + X['Feature3'] * 0.5 + np.random.randn(100)

# -------------------------------------------------------------------------
# 6.1 Correlation-based Selection
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("6.1 Correlation-based Selection")
print("-" * 50)

correlations = X.corr()['Target'].drop('Target').abs().sort_values(ascending=False)
print("Correlation with target:")
print(correlations)

# Select features with correlation > 0.1
selected_features = correlations[correlations > 0.1].index.tolist()
print(f"\nSelected features (corr > 0.1): {selected_features}")

# -------------------------------------------------------------------------
# 6.2 Variance-based Selection
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("6.2 Variance-based Selection")
print("-" * 50)

from sklearn.feature_selection import VarianceThreshold

# Remove low variance features
selector = VarianceThreshold(threshold=0.1)
X_high_variance = selector.fit_transform(X.drop('Target', axis=1))
selected_idx = selector.get_support(indices=True)
selected_cols = X.drop('Target', axis=1).columns[selected_idx]

print(f"Original features: {list(X.drop('Target', axis=1).columns)}")
print(f"Selected (variance > 0.1): {list(selected_cols)}")

# -------------------------------------------------------------------------
# 6.3 Model-based Selection (Feature Importance)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("6.3 Model-based Feature Importance")
print("-" * 50)

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

X_train = X.drop('Target', axis=1)
y_train = X['Target']

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Random Forest Feature Importance:")
print(importance)

# Select important features
selector = SelectFromModel(rf, threshold='mean')
selector.fit(X_train, y_train)
important_features = X_train.columns[selector.get_support()].tolist()
print(f"\nSelected features (importance > mean): {important_features}")

# ============================================================================
# 7. COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("7. COMPLETE PIPELINE EXAMPLE")
print("=" * 70)

def full_feature_engineering_pipeline(df):
    """
    Complete feature engineering pipeline for Titanic-like data.
    """
    df = df.copy()

    # 1. Extract title from name
    def get_title(name):
        title = 'Other'
        if 'Mr.' in name:
            title = 'Mr'
        elif 'Mrs.' in name:
            title = 'Mrs'
        elif 'Miss' in name:
            title = 'Miss'
        elif 'Master' in name:
            title = 'Master'
        return title

    df['Title'] = df['Name'].apply(get_title)

    # 2. Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 3. Cabin features
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')

    # 4. Age imputation and binning
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])

    # 5. Fare binning
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # 6. Fill missing embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    return df

# Apply pipeline
print("Applying full feature engineering pipeline...")
df_processed = full_feature_engineering_pipeline(titanic)

print("\nOriginal columns:")
print(list(titanic.columns))

print("\nNew/Engineered columns:")
engineered_cols = ['Title', 'FamilySize', 'IsAlone', 'HasCabin', 'Deck', 'AgeBin', 'FareBin']
print(engineered_cols)

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Missing Values Strategy
   - Mean/Median: Simple, fast
   - KNN: Considers similarity between samples
   - Forward/Backward: For ordered/time series data

2. Categorical Encoding
   - Label Encoding: Ordinal categories
   - One-Hot Encoding: Nominal categories (low cardinality)
   - Target Encoding: High cardinality (requires CV to avoid leakage)

3. Feature Transformation
   - Log/Sqrt: Reduce skewness
   - Box-Cox: Optimal power transformation
   - Binning: Convert continuous to categorical

4. Feature Creation
   - Extract from text (titles, patterns)
   - Combine features (interactions)
   - Domain-specific features

5. Scaling
   - StandardScaler: Normal distribution
   - MinMaxScaler: Bounded range [0,1]
   - RobustScaler: Outlier resistant

6. Feature Selection
   - Correlation: Quick filter
   - Variance: Remove constant features
   - Model-based: Use algorithm's importance

Next: Model Training (03_model_training.py)
""")
