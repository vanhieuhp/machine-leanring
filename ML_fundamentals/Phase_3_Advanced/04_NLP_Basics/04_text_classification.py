"""
=================================================================
04 - TEXT CLASSIFICATION: Multi-class Document Classification
=================================================================
Topics:
  1. Multi-class text classification pipeline
  2. 20 Newsgroups dataset
  3. Model comparison for text classification
  4. Confusion matrix analysis
  5. Pipeline with hyperparameter tuning
  6. Making predictions on new text
=================================================================
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Load Dataset ──────────────────────────────────────────────────
print("=" * 65)
print("Loading 20 Newsgroups Dataset")
print("=" * 65)

# Use a subset of categories for clarity
categories = [
    'comp.graphics',
    'rec.sport.baseball',
    'sci.med',
    'talk.politics.guns',
]

train_data = fetch_20newsgroups(
    subset='train', categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42,
)
test_data = fetch_20newsgroups(
    subset='test', categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42,
)

X_train, y_train = train_data.data, train_data.target
X_test, y_test = test_data.data, test_data.target
target_names = train_data.target_names

print(f"\n  Categories: {target_names}")
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print(f"\n  Class distribution (train):")
for i, name in enumerate(target_names):
    count = sum(y_train == i)
    print(f"    {name:>25s}: {count:>5d} ({count/len(y_train)*100:.1f}%)")

# Show a sample
print(f"\n  Sample document (first 200 chars):")
print(f"    Category: {target_names[y_train[0]]}")
print(f"    Text: '{X_train[0][:200]}...'")

# ── Section 1: Basic Pipeline ─────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 1: Basic Text Classification Pipeline")
print("=" * 65)

print("""
  Pipeline: Raw Text → TF-IDF → Classifier → Prediction

  Steps:
    1. TfidfVectorizer converts text to numerical features
    2. Classifier learns patterns from features
    3. Predict category of new text
""")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"\n  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

# ── Section 2: Model Comparison ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Model Comparison for Text Classification")
print("=" * 65)

models = {
    "Multinomial NB": MultinomialNB(),
    "Complement NB": ComplementNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SGD Classifier": SGDClassifier(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(max_iter=2000),
}

print(f"\n  {'Model':<22s} {'Test Acc':>10s} {'CV(5)':>10s}")
print("  " + "-" * 44)

best_model_name = ""
best_acc = 0

for name, model in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", model),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()
    print(f"  {name:<22s} {acc:>10.4f} {cv:>10.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name

print(f"\n  🏆 Best model: {best_model_name} ({best_acc:.4f})")

# ── Section 3: Confusion Matrix Analysis ─────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Confusion Matrix Analysis")
print("=" * 65)

# Use the best model (LinearSVC)
best_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", LinearSVC(max_iter=2000)),
])
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
short_names = [n.split('.')[-1] for n in target_names]
print(f"\n  Confusion Matrix:")
print(f"  {'Predicted →':>15s}", end="")
for name in short_names:
    print(f" {name:>10s}", end="")
print()
print("  " + "-" * (16 + len(short_names) * 11))

for i, name in enumerate(short_names):
    print(f"  {name:>15s}", end="")
    for j in range(len(short_names)):
        val = cm[i][j]
        marker = " ✓" if i == j else ""
        print(f" {val:>8d}{marker}", end="")
    print()

# Per-class accuracy
print(f"\n  Per-class accuracy:")
for i, name in enumerate(target_names):
    class_acc = cm[i][i] / cm[i].sum()
    short = name.split('.')[-1]
    print(f"    {short:>15s}: {class_acc:.4f} ({cm[i][i]}/{cm[i].sum()})")

# ── Section 4: Hyperparameter Tuning ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Pipeline Hyperparameter Tuning")
print("=" * 65)

pipe_tune = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000)),
])

param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": [0.1, 1.0, 10.0],
}

print(f"\n  GridSearchCV: {2*2*2*3} combinations × 3-fold")
print("  Running...")

grid = GridSearchCV(pipe_tune, param_grid, cv=3, scoring="accuracy",
                    n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print(f"\n  Best CV Score:  {grid.best_score_:.4f}")
print(f"  Test Accuracy:  {accuracy_score(y_test, grid.predict(X_test)):.4f}")
print(f"  Best Parameters:")
for k, v in grid.best_params_.items():
    print(f"    {k}: {v}")

# ── Section 5: Predicting New Text ────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Predicting New Documents")
print("=" * 65)

new_texts = [
    "The patient was diagnosed with pneumonia and prescribed antibiotics for treatment.",
    "The pitcher threw a fastball strike in the bottom of the ninth inning.",
    "The 3D rendering engine uses ray tracing for realistic lighting effects.",
    "The proposed gun control legislation was debated in the Senate committee.",
    "Heart surgery techniques have improved significantly with new medical technology.",
    "The baseball team won the championship after a dramatic playoff series.",
]

predictions = best_pipe.predict(new_texts)

print(f"\n  {'Predicted Category':>25s}  Text")
print("  " + "-" * 70)
for text, pred in zip(new_texts, predictions):
    cat = target_names[pred].split('.')[-1]
    print(f"  {cat:>25s}  '{text[:55]}...'")

# ── Section 6: Important Features per Category ───────────────────
print("\n" + "=" * 65)
print("SECTION 6: Most Predictive Words per Category")
print("=" * 65)

# Retrain with LogReg to get coefficients
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe_lr.fit(X_train, y_train)

feature_names = pipe_lr.named_steps["tfidf"].get_feature_names_out()
coefs = pipe_lr.named_steps["clf"].coef_

for i, category in enumerate(target_names):
    top_idx = np.argsort(coefs[i])[-8:][::-1]
    top_words = [feature_names[idx] for idx in top_idx]
    short_name = category.split('.')[-1]
    print(f"\n  {short_name}:")
    print(f"    Top words: {', '.join(top_words)}")

# ── Summary ───────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. Pipeline(TfidfVectorizer + Classifier) is the standard approach
  2. LinearSVC and LogReg are usually the best for text classification
  3. Naive Bayes is surprisingly competitive and very fast
  4. N-grams (1,2) capture multi-word features
  5. GridSearchCV can tune both vectorizer and classifier together
  6. Feature analysis reveals what words drive predictions

📋 Text Classification Checklist:
  □ Load and explore data
  □ Build Pipeline (TF-IDF + classifier)
  □ Compare multiple models
  □ Tune hyperparameters with GridSearchCV
  □ Analyze confusion matrix and errors
  □ Check most predictive features per class

📚 Next: exercises.py (NLP Practice Problems)
""")
