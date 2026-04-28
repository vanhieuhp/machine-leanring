"""
=================================================================
NLP BASICS — EXERCISES
=================================================================
5 hands-on exercises with increasing difficulty.
=================================================================
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 1: Text Preprocessing Function (⭐⭐)                ║
# ╚═════════════════════════════════════════════════════════════════╝
print("=" * 65)
print("EXERCISE 1: Build a Text Preprocessing Function")
print("=" * 65)
print("""
📝 Task:
  1. Write a function clean_text(text) that:
     - Converts to lowercase
     - Removes URLs, @mentions, #hashtags
     - Removes punctuation and special chars
     - Removes extra whitespace
  2. Test on the provided sample texts

🎯 Expected: Each sample should be cleaned properly
""")

# === YOUR CODE HERE ===
# def clean_text(text):
#     ...

# === SOLUTION ===
print("--- SOLUTION ---")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text


samples = [
    "AMAZING movie!!! Check https://review.com #best @director 5/5 ⭐⭐⭐",
    "Worst film ever... DON'T watch!! @studio123 #terrible",
    "  It's   okay,  nothing   special   ",
]

for s in samples:
    print(f"  Raw:   '{s}'")
    print(f"  Clean: '{clean_text(s)}'\n")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 2: BoW vs TF-IDF Comparison (⭐⭐)                   ║
# ╚═════════════════════════════════════════════════════════════════╝
print("=" * 65)
print("EXERCISE 2: Compare BoW vs TF-IDF")
print("=" * 65)
print("""
📝 Task:
  1. Use the movie review dataset below
  2. Build two pipelines:
     - CountVectorizer + MultinomialNB
     - TfidfVectorizer + MultinomialNB
  3. Compare accuracy on test set
  4. Which representation works better?

🎯 Expected: TF-IDF should be equal or better
""")

# Dataset
reviews = [
    "This movie was amazing and wonderful", "Great film I loved it",
    "Excellent movie with brilliant acting", "Best movie ever made",
    "Loved the story and characters", "Fantastic performance great movie",
    "Really enjoyed this wonderful film", "Amazing storyline loved it",
    "Beautiful movie excellent direction", "Superb acting great film",
    "Terrible movie worst ever seen", "Awful film hated every minute",
    "Boring and predictable waste time", "Horrible acting bad movie",
    "Disappointed worst film year", "Dreadful movie avoid this garbage",
    "Stupid plot terrible dialogue", "Bad movie poor direction",
    "Waste of time awful storyline", "Painfully boring terrible film",
]
labels = [1]*10 + [0]*10

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.3, random_state=42, stratify=labels
)

# BoW
pipe_bow = Pipeline([("cv", CountVectorizer()), ("nb", MultinomialNB())])
pipe_bow.fit(X_train, y_train)
bow_acc = accuracy_score(y_test, pipe_bow.predict(X_test))

# TF-IDF
pipe_tfidf = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])
pipe_tfidf.fit(X_train, y_train)
tfidf_acc = accuracy_score(y_test, pipe_tfidf.predict(X_test))

print(f"  BoW + NB:    {bow_acc:.4f}")
print(f"  TF-IDF + NB: {tfidf_acc:.4f}")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 3: Document Similarity (⭐⭐)                        ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 3: Find Most Similar Documents")
print("=" * 65)
print("""
📝 Task:
  1. Use TF-IDF to vectorize the documents below
  2. Compute cosine similarity matrix
  3. Find the pair of documents most similar to each query

🎯 Expected: Similar topics should have high cosine similarity
""")

docs = [
    "Python is a great programming language for data science",
    "Machine learning algorithms learn from data",
    "Basketball and football are popular sports in America",
    "Data science uses Python and machine learning",
    "The NBA basketball season starts in October",
]

query = "I love programming in Python for machine learning"

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
tfidf = TfidfVectorizer()
X_docs = tfidf.fit_transform(docs)
X_query = tfidf.transform([query])

similarities = cosine_similarity(X_query, X_docs)[0]

print(f"\n  Query: '{query}'\n")
print(f"  {'Similarity':>12s}  Document")
print("  " + "-" * 60)

ranked = np.argsort(similarities)[::-1]
for idx in ranked:
    print(f"  {similarities[idx]:>12.4f}  '{docs[idx]}'")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 4: Sentiment Analysis Pipeline (⭐⭐⭐)               ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 4: Complete Sentiment Analysis Pipeline")
print("=" * 65)
print("""
📝 Task:
  1. Use the reviews dataset from exercise 2
  2. Create preprocessing function
  3. Build pipeline: preprocess → TF-IDF(bigrams) → LogReg
  4. Train and evaluate
  5. Predict sentiment for 3 new reviews

🎯 Expected: Reasonable predictions on new reviews
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")

# Preprocess all reviews
clean_reviews = [clean_text(r) for r in reviews]
X_train, X_test, y_train, y_test = train_test_split(
    clean_reviews, labels, test_size=0.3, random_state=42, stratify=labels
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)

acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"  Test Accuracy: {acc:.4f}")

new_reviews = [
    "This movie was incredible, I absolutely loved the acting!",
    "What a terrible waste of time, very disappointing film.",
    "The movie was okay, had some good and bad moments.",
]

for review in new_reviews:
    clean = clean_text(review)
    pred = pipe.predict([clean])[0]
    sentiment = "Positive ✅" if pred == 1 else "Negative ❌"
    print(f"  {sentiment}  '{review[:50]}...'")


# ╔═════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 5: 20 Newsgroups Classification (⭐⭐⭐)              ║
# ╚═════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 65)
print("EXERCISE 5: Multi-class Text Classification")
print("=" * 65)
print("""
📝 Task:
  1. Load 20 Newsgroups (categories: sci.space, rec.autos, comp.graphics)
  2. Build 3 pipelines: NB, LogReg, LinearSVC (all with TF-IDF)
  3. Compare using cross-validation
  4. Print classification report for the best model
  5. Predict categories for 3 custom texts

🎯 Expected: Best model accuracy > 0.85
""")

# === YOUR CODE HERE ===
# ...

# === SOLUTION ===
print("--- SOLUTION ---")
from sklearn.datasets import fetch_20newsgroups

categories = ['sci.space', 'rec.autos', 'comp.graphics']
train_data = fetch_20newsgroups(subset='train', categories=categories,
                                 remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories,
                                remove=('headers', 'footers', 'quotes'))

models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(max_iter=2000),
}

print(f"\n  {'Model':<22s} {'Test Acc':>10s}")
print("  " + "-" * 34)

best_name, best_acc, best_pipe = "", 0, None
for name, model in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", model),
    ])
    pipe.fit(train_data.data, train_data.target)
    acc = accuracy_score(test_data.target, pipe.predict(test_data.data))
    print(f"  {name:<22s} {acc:>10.4f}")
    if acc > best_acc:
        best_name, best_acc, best_pipe = name, acc, pipe

print(f"\n  🏆 Best: {best_name} ({best_acc:.4f})")

# Predict custom texts
custom = [
    "NASA launched a new rocket to explore Mars and the solar system.",
    "The new Ford Mustang has an incredible V8 engine and great handling.",
    "The 3D rendering pipeline uses OpenGL shaders for realistic graphics.",
]

preds = best_pipe.predict(custom)
for text, pred in zip(custom, preds):
    cat = categories[pred].split('.')[-1]
    print(f"  [{cat:>10s}]  '{text[:55]}...'")

print("\n✅ NLP exercises complete! Phase 3 Advanced is done! 🎉")
