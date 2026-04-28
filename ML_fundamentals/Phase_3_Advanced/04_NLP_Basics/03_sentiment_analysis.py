"""
=================================================================
03 - SENTIMENT ANALYSIS: Analyzing Text Sentiment with ML
=================================================================
Topics:
  1. Simple rule-based sentiment
  2. ML-based sentiment (TF-IDF + classifiers)
  3. Model comparison (NB, LogReg, SVM)
  4. Error analysis
  5. Handling real-world text challenges
=================================================================
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Create Sample Dataset ─────────────────────────────────────────
# Since we don't have a real dataset, we'll create a realistic one
positive_reviews = [
    "This movie was absolutely amazing! Best film I've seen this year.",
    "Loved every minute of it. Great acting and storyline.",
    "Incredible performance by the lead actor. Highly recommended!",
    "What a masterpiece! The cinematography was breathtaking.",
    "Fantastic movie with excellent direction and acting.",
    "Really enjoyed this film. It was fun and entertaining.",
    "Brilliant storytelling. This movie will stay with you forever.",
    "One of the best movies ever made. A true classic.",
    "Amazing special effects and a compelling story.",
    "Wonderful movie! The characters were so well developed.",
    "Great movie, really fun to watch with amazing visuals.",
    "Superb acting and a thrilling plot. Loved it!",
    "Beautiful film with a touching and emotional story.",
    "Excellent movie! I laughed and cried. Highly recommend.",
    "Perfect blend of action and drama. Thoroughly enjoyed it.",
    "This film exceeded all my expectations. Truly remarkable.",
    "A delightful movie that the whole family can enjoy.",
    "Outstanding performance. This actor deserves an award.",
    "Captivating from start to finish. A must-see movie.",
    "Heartwarming story with brilliant performances all around.",
]

negative_reviews = [
    "Terrible movie. Complete waste of time and money.",
    "Worst film I've ever seen. The acting was awful.",
    "Boring and predictable. I fell asleep halfway through.",
    "Horrible storyline with terrible dialogue.",
    "What a disaster. Don't waste your time on this garbage.",
    "Disappointing movie with poor acting and bad script.",
    "Absolutely dreadful. I want my two hours back.",
    "This movie was painfully bad. Avoid at all costs.",
    "Stupid plot with no character development whatsoever.",
    "A complete mess from start to finish. Awful.",
    "Terrible acting and a confusing storyline. Hated it.",
    "The worst movie of the year. Absolutely unwatchable.",
    "Poor direction and lazy writing. Very disappointing.",
    "Waste of a good cast. The movie was simply terrible.",
    "Boring movie with no redeeming qualities at all.",
    "I couldn't finish watching. It was that bad.",
    "Dull, lifeless, and completely unoriginal.",
    "A huge letdown. The trailer was better than the movie.",
    "Painfully slow and utterly forgettable.",
    "Awful movie with cringe-worthy dialogue throughout.",
]

texts = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Positive: {sum(y_train)} train, {sum(y_test)} test")
print(f"Negative: {len(y_train) - sum(y_train)} train, {len(y_test) - sum(y_test)} test")

# ── Section 1: Rule-Based Sentiment ───────────────────────────────
print("\n" + "=" * 65)
print("SECTION 1: Rule-Based Sentiment Analysis")
print("=" * 65)

positive_words = {"amazing", "great", "excellent", "fantastic", "love", "loved",
                  "wonderful", "brilliant", "incredible", "outstanding", "superb",
                  "beautiful", "perfect", "best", "captivating", "delightful",
                  "heartwarming", "masterpiece", "remarkable", "enjoyed", "fun"}

negative_words = {"terrible", "awful", "horrible", "worst", "boring", "bad",
                  "disappointing", "dreadful", "stupid", "disaster", "poor",
                  "waste", "painful", "garbage", "mess", "unwatchable", "dull",
                  "cringe", "letdown", "forgettable", "hated", "slow"}


def rule_based_sentiment(text):
    """Simple rule-based sentiment analysis."""
    words = set(text.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    
    if pos_count > neg_count:
        return 1  # positive
    elif neg_count > pos_count:
        return 0  # negative
    else:
        return 1  # default to positive

y_pred_rule = [rule_based_sentiment(text) for text in X_test]
rule_acc = accuracy_score(y_test, y_pred_rule)

print(f"\n  Rule-based accuracy: {rule_acc:.4f}")
print("  💡 Simple but limited — can't handle context or unseen words")

# ── Section 2: ML-Based Sentiment ─────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: ML-Based Sentiment Analysis")
print("=" * 65)

# --- 2.1 Naive Bayes ---
print("\n📌 2.1 Naive Bayes + TF-IDF:")
pipe_nb = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", MultinomialNB()),
])
pipe_nb.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, pipe_nb.predict(X_test))
print(f"  Accuracy: {nb_acc:.4f}")

# --- 2.2 Logistic Regression ---
print("\n📌 2.2 Logistic Regression + TF-IDF:")
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe_lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, pipe_lr.predict(X_test))
print(f"  Accuracy: {lr_acc:.4f}")

# --- 2.3 Linear SVM ---
print("\n📌 2.3 Linear SVM + TF-IDF:")
pipe_svm = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LinearSVC()),
])
pipe_svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, pipe_svm.predict(X_test))
print(f"  Accuracy: {svm_acc:.4f}")

# ── Section 3: Model Comparison ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Model Comparison")
print("=" * 65)

print(f"\n  {'Model':<25s} {'Accuracy':>10s}")
print("  " + "-" * 37)
print(f"  {'Rule-based':<25s} {rule_acc:>10.4f}")
print(f"  {'Naive Bayes':<25s} {nb_acc:>10.4f}")
print(f"  {'Logistic Regression':<25s} {lr_acc:>10.4f}")
print(f"  {'Linear SVM':<25s} {svm_acc:>10.4f}")

# ── Section 4: Feature Analysis ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Most Important Words (Feature Analysis)")
print("=" * 65)

# Use LogReg coefficients to find most predictive words
tfidf = pipe_lr.named_steps["tfidf"]
clf = pipe_lr.named_steps["clf"]
feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]

# Top positive words
top_pos_idx = np.argsort(coefs)[-10:][::-1]
top_neg_idx = np.argsort(coefs)[:10]

print("\n  Top 10 POSITIVE indicators:")
for idx in top_pos_idx:
    print(f"    '{feature_names[idx]}': {coefs[idx]:>8.4f}")

print("\n  Top 10 NEGATIVE indicators:")
for idx in top_neg_idx:
    print(f"    '{feature_names[idx]}': {coefs[idx]:>8.4f}")

# ── Section 5: Predicting New Reviews ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Predicting New Reviews")
print("=" * 65)

new_reviews = [
    "This was a really great movie, I enjoyed every second!",
    "Absolutely terrible, wouldn't recommend to anyone.",
    "It was okay, nothing special but not bad either.",
    "The acting was mediocre but the story was compelling.",
    "I've seen better movies. This one was a bit disappointing.",
]

predictions = pipe_lr.predict(new_reviews)
# Get probabilities for confidence
probs = pipe_lr.predict_proba(new_reviews)

print(f"\n  {'Prediction':>10s} {'Confidence':>12s}  Review")
print("  " + "-" * 70)

for review, pred, prob in zip(new_reviews, predictions, probs):
    sentiment = "Positive ✅" if pred == 1 else "Negative ❌"
    confidence = prob.max()
    print(f"  {sentiment:>10s} {confidence:>12.3f}  '{review[:50]}...'")

# ── Section 6: TF-IDF Parameters Impact ──────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: Impact of TF-IDF Parameters")
print("=" * 65)

configs = {
    "Unigrams only": {"ngram_range": (1, 1)},
    "Uni+Bigrams": {"ngram_range": (1, 2)},
    "Top 100 features": {"max_features": 100},
    "Top 500 features": {"max_features": 500},
    "With stopwords": {"stop_words": "english"},
    "sublinear_tf": {"sublinear_tf": True, "ngram_range": (1, 2)},
}

print(f"\n  {'Config':<22s} {'Accuracy':>10s} {'Features':>10s}")
print("  " + "-" * 44)

for name, params in configs.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**params)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    n_feat = pipe.named_steps["tfidf"].transform(X_train).shape[1]
    print(f"  {name:<22s} {acc:>10.4f} {n_feat:>10d}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. ML-based sentiment analysis outperforms rule-based
  2. TF-IDF + LogisticRegression is a strong baseline
  3. Naive Bayes is fast and works well for text
  4. Feature analysis reveals most predictive words
  5. Bigrams capture phrases like "not good" or "really bad"
  6. Use Pipeline to combine preprocessing + model

📊 Best Starter Setup:
  TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
  + LogisticRegression(max_iter=1000)

📚 Next: 04_text_classification.py
""")
