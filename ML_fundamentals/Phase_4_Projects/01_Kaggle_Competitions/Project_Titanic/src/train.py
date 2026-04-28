"""
Training Module
==============

Main training script for Titanic model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle

from data_loader import load_data, combine_train_test, save_submission
from feature_engineering import full_feature_engineering, get_feature_columns
from model import (
    get_baseline_models,
    get_advanced_models,
    get_stacking_ensemble,
    train_and_evaluate_models,
    get_feature_importance
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main(data_dir='data', output_dir='output'):
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("TITANIC SURVIVAL PREDICTION - TRAINING")
    print("=" * 60)

    # Create directories
    Path(output_dir).mkdir(exist_ok=True)

    # ========================================
    # 1. Load Data
    # ========================================
    print("\n[1/6] Loading data...")
    train_df, test_df = load_data(data_dir)

    # ========================================
    # 2. Feature Engineering
    # ========================================
    print("\n[2/6] Feature engineering...")

    # Apply feature engineering
    train_processed, encoding_map = full_feature_engineering(train_df)
    test_processed, _ = full_feature_engineering(test_df)

    # Get feature columns
    feature_cols = [
        'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch',
        'Fare', 'Embarked_encoded', 'FamilySize', 'IsAlone',
        'Title_encoded', 'HasCabin', 'Deck_encoded',
        'IsChild', 'Age*Class', 'FarePerPerson'
    ]

    # Prepare data
    X = train_processed[feature_cols].fillna(0)
    y = train_processed['Survived']

    X_test = test_processed[feature_cols].fillna(0)
    test_ids = test_processed['PassengerId']

    print(f"Training features shape: {X.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Features: {feature_cols}")

    # ========================================
    # 3. Train-Validation Split
    # ========================================
    print("\n[3/6] Creating train-validation split...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    # ========================================
    # 4. Model Training
    # ========================================
    print("\n[4/6] Training models...")

    # Train baseline models
    results = train_and_evaluate_models(X_train, y_train, X_val, y_val)

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]['model']

    print(f"\nBest baseline model: {best_model_name}")
    print(f"CV Accuracy: {results[best_model_name]['cv_mean']:.4f}")

    # ========================================
    # 5. Advanced Models & Ensembling
    # ========================================
    print("\n[5/6] Training advanced models...")

    # Train advanced models
    advanced_models = get_advanced_models()

    for name, model in advanced_models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model.fit(X_train, y_train)
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        print(f"  {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Create stacking ensemble
    print("\nCreating stacking ensemble...")
    base_models = {
        'rf': results['RandomForestTuned']['model'],
        'gb': results['GradientBoostingTuned']['model'],
    }

    # Add available models
    if 'XGBoost' in results:
        base_models['xgb'] = results['XGBoost']['model']
    if 'LightGBM' in results:
        base_models['lgb'] = results['LightGBM']['model']

    stacking_clf = get_stacking_ensemble(base_models)
    stacking_clf.fit(X_train, y_train)

    # Evaluate stacking
    stacking_pred = stacking_clf.predict(X_val)
    stacking_acc = accuracy_score(y_val, stacking_pred)
    print(f"  Stacking CV: {stacking_acc:.4f}")

    # ========================================
    # 6. Final Model Selection
    # ========================================
    print("\n[6/6] Selecting final model...")

    # Compare all models on validation set
    print("\nValidation set comparison:")
    model_scores = []

    for name, result in results.items():
        model = result['model']
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
        else:
            auc = 0

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        model_scores.append({
            'name': name,
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'cv_mean': result['cv_mean']
        })

        print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Add stacking
    y_pred_stacking = stacking_clf.predict(X_val)
    stacking_acc = accuracy_score(y_val, y_pred_stacking)
    stacking_f1 = f1_score(y_val, y_pred_stacking)
    stacking_auc = roc_auc_score(y_val, stacking_clf.predict_proba(X_val)[:, 1])

    model_scores.append({
        'name': 'Stacking',
        'accuracy': stacking_acc,
        'f1': stacking_f1,
        'auc': stacking_auc,
        'cv_mean': stacking_acc
    })

    print(f"  Stacking: Acc={stacking_acc:.4f}, F1={stacking_f1:.4f}, AUC={stacking_auc:.4f}")

    # Select best model
    model_scores_df = pd.DataFrame(model_scores)
    best_overall = model_scores_df.loc[model_scores_df['accuracy'].idxmax()]

    print(f"\nBest model: {best_overall['name']}")
    print(f"Validation Accuracy: {best_overall['accuracy']:.4f}")

    # ========================================
    # Retrain on Full Data and Generate Predictions
    # ========================================
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    # Retrain best models on full training data
    print("\nRetraining on full dataset...")

    # Use stacking or best single model
    if best_overall['name'] == 'Stacking':
        final_model = stacking_clf
    else:
        final_model = results[best_overall['name']]['model']

    # Retrain on full data
    final_model.fit(X, y)

    # Generate predictions
    predictions = final_model.predict(X_test)
    predictions = predictions.astype(int)

    # ========================================
    # Save Results
    # ========================================
    print("\nSaving results...")

    # Save submission
    submission_path = Path(output_dir) / 'submission.csv'
    save_submission(predictions, test_ids, submission_path)

    # Save model
    model_path = Path(output_dir) / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Model saved to: {model_path}")

    # Save scaler
    scaler_path = Path(output_dir) / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save feature importance
    if hasattr(final_model, 'feature_importances_'):
        importance = get_feature_importance(final_model, feature_cols)
        importance_path = Path(output_dir) / 'feature_importance.csv'
        importance.to_csv(importance_path, index=False)
        print(f"Feature importance saved to: {importance_path}")

    # Save model scores
    scores_path = Path(output_dir) / 'model_scores.json'
    model_scores_df.to_json(scores_path, orient='records', indent=2)
    print(f"Model scores saved to: {scores_path}")

    # Save encoding map
    encoding_path = Path(output_dir) / 'encoding_map.json'
    # Convert encoding map to JSON-serializable format
    encoding_dict = {}
    for k, v in encoding_map.items():
        if isinstance(v, dict):
            encoding_dict[k] = {str(k2): int(v2) for k2, v2 in v.items()}
        else:
            encoding_dict[k] = int(v)

    with open(encoding_path, 'w') as f:
        json.dump(encoding_dict, f, indent=2)
    print(f"Encoding map saved to: {encoding_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return final_model, results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Titanic model')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Path to output directory')

    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
