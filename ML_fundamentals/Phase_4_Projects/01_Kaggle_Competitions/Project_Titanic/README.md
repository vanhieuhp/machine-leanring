# Titanic Survival Prediction Project

## Project Overview

This project implements a complete machine learning pipeline to predict passenger survival on the Titanic dataset.

## Objective

Build a model to predict whether a passenger survived the Titanic disaster based on their demographic and travel information.

## Dataset

- **Source**: Kaggle Titanic Competition
- **Features**: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- **Target**: Survived (0 = No, 1 = Yes)

## Project Structure

```
Project_Titanic/
├── README.md
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── notebooks/
│   └── exploration.ipynb
└── submission/
    └── submission.csv
```

## Approach

### 1. Data Exploration
- Understand data distributions
- Identify missing values
- Analyze feature correlations with survival

### 2. Feature Engineering
- Extract titles from names
- Create family size features
- Extract cabin deck information
- Handle missing values appropriately

### 3. Model Training
- Baseline: Logistic Regression
- Tree-based: Decision Tree, Random Forest
- Boosting: Gradient Boosting, XGBoost
- Ensemble: Voting, Stacking

### 4. Evaluation
- Cross-validation (Stratified K-Fold)
- Metrics: Accuracy, F1-Score, ROC-AUC

## Getting Started

1. Download data from Kaggle:
   ```
   https://www.kaggle.com/c/titanic/data
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run training:
   ```
   python src/train.py
   ```

4. Generate predictions:
   ```
   python src/predict.py
   ```

## Results

| Model | CV Accuracy | CV F1 |
|-------|-------------|-------|
| Logistic Regression | ~0.78 | ~0.70 |
| Decision Tree | ~0.80 | ~0.73 |
| Random Forest | ~0.83 | ~0.76 |
| Gradient Boosting | ~0.84 | ~0.78 |
| XGBoost | ~0.85 | ~0.79 |
| Ensemble (Stacking) | ~0.86 | ~0.80 |

## Key Insights

1. **Sex** is the most important feature - females had much higher survival rate
2. **Pclass** matters - 1st class passengers had higher survival rates
3. **Family size** affects survival - small families had better outcomes
4. **Title** (extracted from name) provides useful demographic information

## Next Steps

- Try more advanced feature engineering
- Experiment with deep learning
- Tune hyperparameters more extensively
- Try neural network approaches
