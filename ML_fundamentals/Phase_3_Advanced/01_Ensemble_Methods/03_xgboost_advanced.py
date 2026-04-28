"""
=================================================================
03 - XGBoost & LightGBM: Production-Grade Gradient Boosting
=================================================================
Topics: XGBoost basics, early stopping, hyperparameter tuning,
        LightGBM, comparison, imbalanced data handling
=================================================================
Prerequisites: pip install xgboost lightgbm
=================================================================
"""
import numpy as np
import time
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

# Import XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
    print("✅ XGBoost imported")
except ImportError:
    HAS_XGB = False
    print("❌ XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✅ LightGBM imported")
except ImportError:
    HAS_LGB = False
    print("❌ LightGBM not installed. Run: pip install lightgbm")

# ── Dataset ──────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Section 1: XGBoost Basics ────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 1: XGBoost Basics")
print("=" * 65)

if HAS_XGB:
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, eval_metric="logloss",
    )
    xgb_clf.fit(X_train, y_train)
    print(f"\n  Train Accuracy: {accuracy_score(y_train, xgb_clf.predict(X_train)):.4f}")
    print(f"  Test Accuracy:  {accuracy_score(y_test, xgb_clf.predict(X_test)):.4f}")
    print(f"  AUC-ROC:        {roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1]):.4f}")

# ── Section 2: Early Stopping ────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Early Stopping")
print("=" * 65)

if HAS_XGB:
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    xgb_early = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        random_state=42, eval_metric="logloss", early_stopping_rounds=20,
    )
    xgb_early.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"\n  Best iteration: {xgb_early.best_iteration}")
    print(f"  Test Accuracy:  {accuracy_score(y_test, xgb_early.predict(X_test)):.4f}")

    print("\n  Without early stopping (overfitting demo):")
    for n_est in [50, 100, 500, 1000]:
        clf = xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.05,
                                 max_depth=5, random_state=42, eval_metric="logloss")
        clf.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        te_acc = accuracy_score(y_test, clf.predict(X_test))
        flag = '⚠️' if tr_acc - te_acc > 0.03 else '✅'
        print(f"    n={n_est:>4d}: train={tr_acc:.4f}, test={te_acc:.4f} {flag}")

# ── Section 3: Hyperparameter Tuning ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: XGBoost Hyperparameter Tuning")
print("=" * 65)

if HAS_XGB:
    print("""
    Key hyperparameters:
    ─────────────────────────────────────────────────
    learning_rate (eta)   : Step size, 0.01–0.3
    max_depth             : Tree depth, 3–10
    n_estimators          : Rounds, 100–5000
    subsample             : Row sampling, 0.5–1.0
    colsample_bytree      : Feature sampling, 0.3–1.0
    reg_alpha             : L1 regularization, 0–10
    reg_lambda            : L2 regularization, 0–10
    gamma                 : Min loss for split, 0–5
    min_child_weight      : Min weight in child, 1–10
    scale_pos_weight      : For imbalanced classes
    """)

    param_dist = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }
    rs = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        param_dist, n_iter=20, cv=3, scoring="accuracy",
        random_state=42, n_jobs=-1, verbose=0,
    )
    print("  Running RandomizedSearchCV (20 iterations)...")
    rs.fit(X_train, y_train)
    print(f"  Best CV Score:  {rs.best_score_:.4f}")
    print(f"  Test Accuracy:  {accuracy_score(y_test, rs.predict(X_test)):.4f}")
    print(f"  Best Params:    {rs.best_params_}")

# ── Section 4: LightGBM ──────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: LightGBM")
print("=" * 65)

if HAS_LGB:
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        num_leaves=31, random_state=42, verbose=-1,
    )
    lgb_clf.fit(X_train, y_train)
    print(f"\n  Train Accuracy: {accuracy_score(y_train, lgb_clf.predict(X_train)):.4f}")
    print(f"  Test Accuracy:  {accuracy_score(y_test, lgb_clf.predict(X_test)):.4f}")

    print("""
    LightGBM-specific params:
      num_leaves     : Controls complexity (default=31), should be <= 2^max_depth
      min_data_in_leaf: Min samples per leaf (default=20)
      max_bin        : Bins for features (default=255)
      feature_fraction: Like colsample_bytree
      bagging_fraction: Like subsample
    """)

# ── Section 5: Speed Comparison ───────────────────────────────────
print("=" * 65)
print("SECTION 5: Speed Comparison")
print("=" * 65)

X_big, y_big = make_classification(n_samples=5000, n_features=50, random_state=42)
X_b_tr, X_b_te, y_b_tr, y_b_te = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

results = {}
start = time.time()
GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42).fit(X_b_tr, y_b_tr)
results["sklearn"] = time.time() - start

if HAS_XGB:
    start = time.time()
    xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42, eval_metric="logloss").fit(X_b_tr, y_b_tr)
    results["XGBoost"] = time.time() - start

if HAS_LGB:
    start = time.time()
    lgb.LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, verbose=-1).fit(X_b_tr, y_b_tr)
    results["LightGBM"] = time.time() - start

base = results["sklearn"]
print(f"\n  {'Model':<12s} {'Time':>8s} {'Speedup':>10s}")
print("  " + "-" * 32)
for name, t in results.items():
    print(f"  {name:<12s} {t:>7.3f}s {base/t:>9.1f}x")

# ── Section 6: Imbalanced Data ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Handling Imbalanced Data")
print("=" * 65)

X_imb, y_imb = make_classification(n_samples=2000, n_features=20,
    weights=[0.9, 0.1], random_state=42)
X_i_tr, X_i_te, y_i_tr, y_i_te = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)
print(f"\n  Class distribution: {np.bincount(y_i_tr)}")

if HAS_XGB:
    xgb_unbal = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    xgb_unbal.fit(X_i_tr, y_i_tr)
    f1_unbal = f1_score(y_i_te, xgb_unbal.predict(X_i_te))

    ratio = np.sum(y_i_tr == 0) / np.sum(y_i_tr == 1)
    xgb_bal = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=ratio,
                                 random_state=42, eval_metric="logloss")
    xgb_bal.fit(X_i_tr, y_i_tr)
    f1_bal = f1_score(y_i_te, xgb_bal.predict(X_i_te))

    print(f"  Without balancing:     F1 = {f1_unbal:.4f}")
    print(f"  With scale_pos_weight: F1 = {f1_bal:.4f}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. XGBoost adds L1/L2 regularization to gradient boosting
  2. LightGBM is faster (leaf-wise growth, histograms)
  3. ALWAYS use early stopping with boosting
  4. Tune: learning_rate → tree params → regularization
  5. Use scale_pos_weight for imbalanced data

📚 Next: 04_stacking.py (Stacking & Voting Ensembles)
""")
