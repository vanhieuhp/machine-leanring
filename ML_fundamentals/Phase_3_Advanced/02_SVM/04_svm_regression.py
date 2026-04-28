"""
=================================================================
04 - SVM FOR REGRESSION (SVR)
=================================================================
Topics:
  1. SVR concept (epsilon-tube)
  2. Linear SVR
  3. RBF SVR
  4. Epsilon and C parameter effects
  5. Comparison with other regressors
  6. Hyperparameter tuning for SVR
=================================================================
"""

import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

# ── Section 1: SVR Concept ───────────────────────────────────────
print("=" * 65)
print("SECTION 1: SVR — The Epsilon-Tube Concept")
print("=" * 65)

print("""
  SVR fits a regression line with an ε-tube around it:

        ○                    ← penalized (outside tube)
      ___○__________________
     |  ○  ε-tube           |
     |   ──────────────     |  ← regression line
     |        ○  ○          |
     |________________________|
              ○              ← penalized (outside tube)

  • Points INSIDE the tube: no penalty (0 loss)
  • Points OUTSIDE the tube: penalized proportional to distance
  • ε controls tube width
  • C controls penalty for points outside tube
""")

# ── Section 2: Basic SVR ─────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Basic SVR")
print("=" * 65)

# Create regression dataset
X, y = make_regression(
    n_samples=500, n_features=10, n_informative=5,
    noise=20, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVR
pipe_linear = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="linear", C=1.0))])
pipe_linear.fit(X_train, y_train)
y_pred_lin = pipe_linear.predict(X_test)

# RBF SVR
pipe_rbf = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0))])
pipe_rbf.fit(X_train, y_train)
y_pred_rbf = pipe_rbf.predict(X_test)

print(f"\n  {'Kernel':<10s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s}")
print("  " + "-" * 40)
for name, y_pred in [("Linear", y_pred_lin), ("RBF", y_pred_rbf)]:
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  {name:<10s} {r2:>8.4f} {rmse:>10.4f} {mae:>10.4f}")

# ── Section 3: Epsilon Parameter ──────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: The Epsilon (ε) Parameter")
print("=" * 65)

print("""
  ε controls the width of the no-penalty tube:
    • Large ε → wide tube → fewer support vectors → simpler model
    • Small ε → narrow tube → more support vectors → complex model
""")

print(f"\n  {'epsilon':>10s} {'R²':>8s} {'RMSE':>10s} {'# SVs':>8s}")
print("  " + "-" * 40)

for eps in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
    pipe = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, epsilon=eps))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    n_sv = len(pipe.named_steps["svr"].support_)
    print(f"  {eps:>10.2f} {r2:>8.4f} {rmse:>10.4f} {n_sv:>8d}")

print("\n  💡 Default ε=0.1 works well for most cases")

# ── Section 4: C Parameter in SVR ────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: C Parameter in SVR")
print("=" * 65)

print(f"\n  {'C':>10s} {'R²':>8s} {'RMSE':>10s} {'# SVs':>8s}")
print("  " + "-" * 40)

for C in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    pipe = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="rbf", C=C))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    n_sv = len(pipe.named_steps["svr"].support_)
    print(f"  {C:>10.2f} {r2:>8.4f} {rmse:>10.4f} {n_sv:>8d}")

# ── Section 5: Comparison with Other Regressors ──────────────────
print("\n" + "=" * 65)
print("SECTION 5: SVR vs Other Regressors")
print("=" * 65)

# Use diabetes dataset
diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
    "SVR (Linear)": Pipeline([("s", StandardScaler()), ("m", SVR(kernel="linear", C=1.0))]),
    "SVR (RBF)": Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", C=10.0))]),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boost": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print(f"\n  Diabetes Dataset Comparison:")
print(f"  {'Model':<22s} {'R²':>8s} {'RMSE':>10s} {'CV(5) R²':>10s}")
print("  " + "-" * 52)

for name, model in models.items():
    model.fit(X_d_train, y_d_train)
    y_pred = model.predict(X_d_test)
    r2 = r2_score(y_d_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_d_test, y_pred))
    cv = cross_val_score(model, X_d_train, y_d_train, cv=5, scoring="r2").mean()
    print(f"  {name:<22s} {r2:>8.4f} {rmse:>10.4f} {cv:>10.4f}")

# ── Section 6: Hyperparameter Tuning ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: GridSearchCV for SVR")
print("=" * 65)

param_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__gamma": ["scale", 0.001, 0.01, 0.1],
    "svr__epsilon": [0.01, 0.1, 0.5],
}

pipe = Pipeline([("s", StandardScaler()), ("svr", SVR(kernel="rbf"))])
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_d_train, y_d_train)

print(f"\n  Best Parameters: {grid.best_params_}")
print(f"  Best CV R²:      {grid.best_score_:.4f}")
print(f"  Test R²:         {r2_score(y_d_test, grid.predict(X_d_test)):.4f}")
print(f"  Test RMSE:       {np.sqrt(mean_squared_error(y_d_test, grid.predict(X_d_test))):.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. SVR creates an ε-tube around the regression line
  2. ε controls the tube width (tolerance for errors)
  3. C controls penalty for points outside the tube
  4. Always scale features for SVR!
  5. SVR works well for small-medium datasets
  6. For large datasets, tree-based methods are usually better

📚 Next: exercises.py (SVM Practice Problems)
""")
