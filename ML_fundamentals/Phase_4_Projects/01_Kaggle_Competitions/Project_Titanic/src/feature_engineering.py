"""
Feature Engineering Module
=========================

Handles all feature engineering for Titanic dataset.
"""

import pandas as pd
import numpy as np


def extract_title(name):
    """
    Extract title from passenger name.

    Parameters:
    -----------
    name : str
        Passenger name

    Returns:
    --------
    str
        Title (Mr, Mrs, Miss, Master, Other)
    """
    title = 'Other'
    if 'Mr.' in name:
        title = 'Mr'
    elif 'Mrs.' in name:
        title = 'Mrs'
    elif 'Miss' in name:
        title = 'Miss'
    elif 'Master' in name:
        title = 'Master'
    elif 'Dr.' in name:
        title = 'Dr'
    elif 'Rev.' in name:
        title = 'Rev'
    # Countesses, Dons, etc.
    elif 'Countess' in name or 'Lady' in name or 'Sir' in name:
        title = 'Nobility'
    elif 'Ms.' in name or 'Mlle' in name or 'Mme' in name:
        title = 'Miss'  # Normalize

    return title


def extract_deck(cabin):
    """
    Extract deck letter from cabin number.

    Parameters:
    -----------
    cabin : str
        Cabin number (e.g., 'C85')

    Returns:
    --------
    str
        Deck letter or 'Unknown'
    """
    if pd.isna(cabin):
        return 'Unknown'
    return cabin[0]


def create_family_features(df):
    """
    Create family-related features.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe

    Returns:
    --------
    df : DataFrame
        Dataframe with family features
    """
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for self

    # Is alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Family category
    def get_family_category(size):
        if size == 1:
            return 'Alone'
        elif size <= 4:
            return 'Small'
        else:
            return 'Large'

    df['FamilyCategory'] = df['FamilySize'].apply(get_family_category)

    return df


def create_age_features(df):
    """
    Create age-related features.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe

    Returns:
    --------
    df : DataFrame
        Dataframe with age features
    """
    # Fill missing age with median by title (better imputation)
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill any remaining NaN with overall median
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Age binning
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Young', 'Middle', 'Senior']
    )

    # Is child
    df['IsChild'] = (df['Age'] < 16).astype(int)

    return df


def create_fare_features(df):
    """
    Create fare-related features.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe

    Returns:
    --------
    df : DataFrame
        Dataframe with fare features
    """
    # Fill missing fare with median by class
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fare per person (for families)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Fare binning
    df['FareBin'] = pd.qcut(
        df['Fare'].clip(lower=0.01),  # Avoid log(0)
        q=4,
        labels=['Low', 'Medium', 'High', 'VeryHigh'],
        duplicates='drop'
    )

    # Log fare (handle skewness)
    df['LogFare'] = np.log1p(df['Fare'])

    return df


def create_cabin_features(df):
    """
    Create cabin-related features.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe

    Returns:
    --------
    df : DataFrame
        Dataframe with cabin features
    """
    # Has cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Deck
    df['Deck'] = df['Cabin'].apply(extract_deck)

    # Number of cabins
    df['NumCabins'] = df['Cabin'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    return df


def create_interaction_features(df):
    """
    Create interaction features.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe

    Returns:
    --------
    df : DataFrame
        Dataframe with interaction features
    """
    # Age * Class
    df['Age*Class'] = df['Age'] * df['Pclass']

    # Sex * Class
    df['Sex_Class'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)

    # Title * Class
    df['Title_Class'] = df['Title'].astype(str) + '_' + df['Pclass'].astype(str)

    return df


def encode_categorical(df, encoding_map=None):
    """
    Encode categorical variables.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    encoding_map : dict, optional
        Pre-defined encoding map

    Returns:
    --------
    df : DataFrame
        Encoded dataframe
    encoding_map : dict
        Encoding maps used
    """
    if encoding_map is None:
        encoding_map = {}

    # Sex encoding
    if 'Sex' in df.columns:
        if 'Sex' not in encoding_map:
            encoding_map['Sex'] = {'male': 0, 'female': 1}
        df['Sex_encoded'] = df['Sex'].map(encoding_map['Sex'])

    # Embarked encoding
    if 'Embarked' in df.columns:
        if 'Embarked' not in encoding_map:
            encoding_map['Embarked'] = {'S': 0, 'C': 1, 'Q': 2}
        df['Embarked_encoded'] = df['Embarked'].map(encoding_map['Embarked'])

    # Title encoding
    if 'Title' in df.columns:
        if 'Title' not in encoding_map:
            title_order = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
                          'Dr': 4, 'Rev': 5, 'Nobility': 6, 'Other': 7}
            encoding_map['Title'] = title_order
        df['Title_encoded'] = df['Title'].map(encoding_map['Title'])

    # Family category encoding
    if 'FamilyCategory' in df.columns:
        if 'FamilyCategory' not in encoding_map:
            encoding_map['FamilyCategory'] = {'Alone': 0, 'Small': 1, 'Large': 2}
        df['FamilyCategory_encoded'] = df['FamilyCategory'].map(
            encoding_map['FamilyCategory']
        )

    # Deck encoding
    if 'Deck' in df.columns:
        if 'Deck' not in encoding_map:
            # Order decks from front to back of ship
            deck_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                         'F': 5, 'G': 6, 'T': 7, 'Unknown': 8}
            encoding_map['Deck'] = deck_order
        df['Deck_encoded'] = df['Deck'].map(encoding_map['Deck'])

    return df, encoding_map


def full_feature_engineering(df):
    """
    Apply all feature engineering steps.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe

    Returns:
    --------
    df : DataFrame
        Feature-engineered dataframe
    encoding_map : dict
        Encoding maps used
    """
    df = df.copy()

    # Extract title from name
    df['Title'] = df['Name'].apply(extract_title)

    # Create family features
    df = create_family_features(df)

    # Create age features
    df = create_age_features(df)

    # Create fare features
    df = create_fare_features(df)

    # Create cabin features
    df = create_cabin_features(df)

    # Create interaction features
    df = create_interaction_features(df)

    # Encode categorical
    df, encoding_map = encode_categorical(df)

    return df, encoding_map


def get_feature_columns():
    """
    Get list of feature columns for modeling.

    Returns:
    --------
    list
        List of feature column names
    """
    feature_cols = [
        'Pclass',
        'Sex_encoded',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked_encoded',
        'FamilySize',
        'IsAlone',
        'Title_encoded',
        'AgeBin_encoded',
        'FarePerPerson',
        'HasCabin',
        'Deck_encoded',
        'IsChild',
        'Age*Class',
    ]

    return feature_cols


if __name__ == '__main__':
    # Test feature engineering
    print("Testing feature engineering...")
    print("This is a module - import it to use")
