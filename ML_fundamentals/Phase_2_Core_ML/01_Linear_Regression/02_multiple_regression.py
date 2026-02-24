"""
Linear Regression - Part 2: Multiple Regression
================================================

This module covers:
- Multiple features
- Feature scaling
- Coefficient interpretation
- Multicollinearity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================================
# 1. MULTIPLE LINEAR REGRESSION
# ============================================================================

print("=" * 70)
print("1. MULTIPLE LINEAR REGRESSION")
print("=" * 70)

# Create sample data with multiple features
np.random.seed(42)
n_samples = 100

# Features: age, experience, education
age = np.random.randint(25, 65, n_samples)
experience = np.random.randint(0, 40, n_samples)
education = np.random.randint(12, 20, n_samples)

# Target: salary (depends on all features)
salary = 30000 + 500*age + 1000*experience + 2000*education + np.random.randn(n_samples) * 5000

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'experience': experience,
    'education': education,
    'salary': salary
})

print("Data shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

print("\nStatistics:")
print(data.describe())

# ============================================================================
# 2. TRAIN MODEL WITHOUT SCALING
# ============================================================================

print("\n" + "=" * 70)
print("2. TRAIN MODEL WITHOUT SCALING")
print("=" * 70)

X = data[['age', 'experience', 'education']]
y = data['salary']

model = LinearRegression()
model.fit(X, y)

# Coefficients
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# Predictions
y_pred = model.predict(X)

# Evaluation
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print(f"\nMetrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================

print("\n" + "=" * 70)
print("3. FEATURE SCALING")
print("=" * 70)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original X (first 5 rows):")
print(X.head())

print("\nScaled X (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=X.columns).head())

# Train model with scaled features
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

print("\nCoefficients (scaled features):")
for feature, coef in zip(X.columns, model_scaled.coef_):
    print(f"  {feature}: {coef:.4f}")

# Predictions
y_pred_scaled = model_scaled.predict(X_scaled)

# Evaluation
r2_scaled = r2_score(y, y_pred_scaled)
print(f"\nR² Score (scaled): {r2_scaled:.4f}")

# ============================================================================
# 4. COEFFICIENT INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("4. COEFFICIENT INTERPRETATION")
print("=" * 70)

print("Interpretation of coefficients:")
print(f"  Age: For each additional year of age, salary increases by ${model.coef_[0]:.2f}")
print(f"  Experience: For each additional year of experience, salary increases by ${model.coef_[1]:.2f}")
print(f"  Education: For each additional year of education, salary increases by ${model.coef_[2]:.2f}")

# ============================================================================
# 5. PREDICTIONS ON NEW DATA
# ============================================================================

print("\n" + "=" * 70)
print("5. PREDICTIONS ON NEW DATA")
print("=" * 70)

# New person
new_person = pd.DataFrame({
    'age': [35],
    'experience': [10],
    'education': [16]
})

predicted_salary = model.predict(new_person)[0]
print(f"New person: Age={new_person['age'][0]}, Experience={new_person['experience'][0]}, Education={new_person['education'][0]}")
print(f"Predicted salary: ${predicted_salary:.2f}")

# ============================================================================
# 6. RESIDUAL ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("6. RESIDUAL ANALYSIS")
print("=" * 70)

residuals = y - y_pred

print(f"Residuals statistics:")
print(f"  Mean: {np.mean(residuals):.4f} (should be close to 0)")
print(f"  Std Dev: {np.std(residuals):.4f}")
print(f"  Min: {np.min(residuals):.4f}")
print(f"  Max: {np.max(residuals):.4f}")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("7. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y, y_pred, alpha=0.5)
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Salary')
axes[0, 0].set_ylabel('Predicted Salary')
axes[0, 0].set_title('Actual vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Salary')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals distribution
axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residuals Distribution')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Feature importance (absolute coefficients)
feature_importance = np.abs(model.coef_)
axes[1, 1].barh(X.columns, feature_importance, color='steelblue')
axes[1, 1].set_xlabel('Absolute Coefficient Value')
axes[1, 1].set_title('Feature Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ============================================================================
# 8. PRACTICAL EXAMPLE: Student Performance
# ============================================================================

print("\n" + "=" * 70)
print("8. PRACTICAL EXAMPLE: Student Performance")
print("=" * 70)

# Create student data
np.random.seed(42)
n_students = 50

study_hours = np.random.uniform(1, 10, n_students)
sleep_hours = np.random.uniform(4, 10, n_students)
attendance = np.random.uniform(0.5, 1.0, n_students)

# GPA depends on all factors
gpa = 2.0 + 0.3*study_hours + 0.2*sleep_hours + 1.5*attendance + np.random.randn(n_students) * 0.3
gpa = np.clip(gpa, 0, 4.0)  # GPA between 0 and 4

student_data = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'attendance': attendance,
    'gpa': gpa
})

# Train model
X_students = student_data[['study_hours', 'sleep_hours', 'attendance']]
y_students = student_data['gpa']

model_students = LinearRegression()
model_students.fit(X_students, y_students)

# Evaluate
y_pred_students = model_students.predict(X_students)
r2_students = r2_score(y_students, y_pred_students)

print(f"Model: GPA = {model_students.coef_[0]:.4f}*study + {model_students.coef_[1]:.4f}*sleep + {model_students.coef_[2]:.4f}*attendance + {model_students.intercept_:.4f}")
print(f"R² Score: {r2_students:.4f}")

# Predict for new student
new_student = pd.DataFrame({
    'study_hours': [7],
    'sleep_hours': [8],
    'attendance': [0.95]
})

predicted_gpa = model_students.predict(new_student)[0]
print(f"\nNew student: Study={new_student['study_hours'][0]}h, Sleep={new_student['sleep_hours'][0]}h, Attendance={new_student['attendance'][0]:.0%}")
print(f"Predicted GPA: {predicted_gpa:.2f}")
