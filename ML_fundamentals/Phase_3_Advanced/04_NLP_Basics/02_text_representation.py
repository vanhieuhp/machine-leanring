"""
=================================================================
02 - TEXT REPRESENTATION: BoW, TF-IDF, N-grams
=================================================================
Topics:
  1. Bag of Words (CountVectorizer)
  2. TF-IDF (TfidfVectorizer)
  3. N-grams
  4. Feature extraction parameters
  5. Comparing representations
  6. Hashing Vectorizer
=================================================================
"""

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, HashingVectorizer
)
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Sample documents
documents = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love deep learning and machine learning",
    "Natural language processing is part of machine learning",
    "Deep learning is a subset of machine learning",
]

# ── Section 1: Bag of Words (CountVectorizer) ────────────────────
print("=" * 65)
print("SECTION 1: Bag of Words (CountVectorizer)")
print("=" * 65)

print("""
  Bag of Words: count word occurrences in each document
  Each document becomes a vector of word counts
""")

# Basic CountVectorizer
cv = CountVectorizer()
X_bow = cv.fit_transform(documents)

print(f"\n  Vocabulary: {cv.get_feature_names_out()}")
print(f"  Shape: {X_bow.shape} (documents × vocabulary)")
print(f"\n  Document-Term Matrix:")

vocab = cv.get_feature_names_out()
print(f"  {'Doc':>5s}", end="")
for word in vocab:
    print(f" {word:>6s}", end="")
print()
print("  " + "-" * (6 + len(vocab) * 7))

for i, doc in enumerate(documents):
    print(f"  {i+1:>5d}", end="")
    for j in range(len(vocab)):
        print(f" {X_bow[i, j]:>6d}", end="")
    print(f"  | '{doc[:35]}...'")

# ── Section 2: TF-IDF ────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: TF-IDF (Term Frequency × Inverse Document Frequency)")
print("=" * 65)

print("""
  TF(word, doc) = count(word in doc) / total words in doc
  IDF(word) = log(total docs / docs containing word) + 1
  TF-IDF = TF × IDF

  Words that appear in MANY documents get LOW weight (e.g., "is", "the")
  Words unique to few documents get HIGH weight (e.g., "quantum")
""")

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)

print(f"\n  TF-IDF Matrix (rounded):")
vocab_tf = tfidf.get_feature_names_out()
print(f"  {'Doc':>5s}", end="")
for word in vocab_tf:
    print(f" {word:>8s}", end="")
print()
print("  " + "-" * (6 + len(vocab_tf) * 9))

for i in range(len(documents)):
    print(f"  {i+1:>5d}", end="")
    for j in range(len(vocab_tf)):
        val = X_tfidf[i, j]
        print(f" {val:>8.3f}", end="")
    print()

# Show which words are most/least important
print("\n  Weight analysis:")
mean_tfidf = np.array(X_tfidf.mean(axis=0)).flatten()
sorted_idx = np.argsort(mean_tfidf)

print("  Least important (common words):")
for i in range(min(3, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"    '{vocab_tf[idx]}': avg TF-IDF = {mean_tfidf[idx]:.4f}")

print("  Most important (unique words):")
for i in range(1, min(4, len(sorted_idx) + 1)):
    idx = sorted_idx[-i]
    print(f"    '{vocab_tf[idx]}': avg TF-IDF = {mean_tfidf[idx]:.4f}")

# ── Section 3: N-grams ───────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: N-grams")
print("=" * 65)

print("""
  N-grams capture word sequences (context):
    Unigram (1): ["I", "love", "machine", "learning"]
    Bigram  (2): ["I love", "love machine", "machine learning"]
    Trigram (3): ["I love machine", "love machine learning"]
""")

# Unigrams only
cv_uni = CountVectorizer(ngram_range=(1, 1))
X_uni = cv_uni.fit_transform(documents)
print(f"\n  Unigrams:  {X_uni.shape[1]} features")
print(f"    Examples: {list(cv_uni.get_feature_names_out()[:5])}")

# Bigrams only
cv_bi = CountVectorizer(ngram_range=(2, 2))
X_bi = cv_bi.fit_transform(documents)
print(f"\n  Bigrams:   {X_bi.shape[1]} features")
print(f"    Examples: {list(cv_bi.get_feature_names_out()[:5])}")

# Uni + Bigrams (most common choice)
cv_unbi = CountVectorizer(ngram_range=(1, 2))
X_unbi = cv_unbi.fit_transform(documents)
print(f"\n  Uni+Bi:    {X_unbi.shape[1]} features")
print(f"    Examples: {list(cv_unbi.get_feature_names_out()[:8])}")

# Trigrams
cv_tri = CountVectorizer(ngram_range=(1, 3))
X_tri = cv_tri.fit_transform(documents)
print(f"\n  Uni+Bi+Tri: {X_tri.shape[1]} features")

print("\n  💡 (1,2) is the most common choice — captures phrases")
print("     but (1,3) can get very large quickly!")

# ── Section 4: TfidfVectorizer Parameters ─────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Important Vectorizer Parameters")
print("=" * 65)

print("""
  ┌─────────────────┬──────────────────────────────────────────┐
  │ Parameter        │ Description                              │
  ├─────────────────┼──────────────────────────────────────────┤
  │ max_features    │ Keep only top N most frequent words       │
  │ max_df          │ Ignore words in >X% of docs (removes     │
  │                 │ too common words, like custom stopwords)  │
  │ min_df          │ Ignore words in <X docs (removes rare)   │
  │ ngram_range     │ (min_n, max_n) for n-grams               │
  │ stop_words      │ 'english' or custom list                  │
  │ sublinear_tf    │ Use log(1+tf) instead of tf               │
  │ norm            │ 'l2' (default), 'l1', or None             │
  └─────────────────┴──────────────────────────────────────────┘
""")

# Demo: different configurations
configs = {
    "Default": TfidfVectorizer(),
    "Max 10 features": TfidfVectorizer(max_features=10),
    "Bigrams": TfidfVectorizer(ngram_range=(1, 2)),
    "No stopwords": TfidfVectorizer(stop_words='english'),
    "min_df=2": TfidfVectorizer(min_df=2),
    "sublinear_tf": TfidfVectorizer(sublinear_tf=True),
}

print(f"  {'Config':<20s} {'Features':>10s}")
print("  " + "-" * 32)
for name, vec in configs.items():
    X = vec.fit_transform(documents)
    print(f"  {name:<20s} {X.shape[1]:>10d}")

# ── Section 5: Document Similarity ────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Document Similarity (Cosine Similarity)")
print("=" * 65)

print("""
  Cosine similarity measures how similar two documents are.
  cos(A, B) = (A · B) / (||A|| × ||B||)
  Range: 0 (completely different) to 1 (identical)
""")

tfidf_sim = TfidfVectorizer()
X_sim = tfidf_sim.fit_transform(documents)

sim_matrix = cosine_similarity(X_sim)

print(f"\n  Document Similarity Matrix:")
print(f"  {'':>5s}", end="")
for i in range(len(documents)):
    print(f" {'D'+str(i+1):>6s}", end="")
print()
print("  " + "-" * (6 + len(documents) * 7))

for i in range(len(documents)):
    print(f"  {'D'+str(i+1):>5s}", end="")
    for j in range(len(documents)):
        print(f" {sim_matrix[i][j]:>6.3f}", end="")
    print()

# Find most similar pair
max_sim = 0
max_pair = (0, 0)
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        if sim_matrix[i][j] > max_sim:
            max_sim = sim_matrix[i][j]
            max_pair = (i, j)

print(f"\n  Most similar pair: D{max_pair[0]+1} & D{max_pair[1]+1} (similarity={max_sim:.3f})")
print(f"    D{max_pair[0]+1}: '{documents[max_pair[0]]}'")
print(f"    D{max_pair[1]+1}: '{documents[max_pair[1]]}'")

# ── Section 6: HashingVectorizer ──────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: HashingVectorizer (for large datasets)")
print("=" * 65)

print("""
  HashingVectorizer uses hashing trick:
    ✅ Fixed memory size (no vocabulary stored)
    ✅ Can handle streaming data
    ✅ Very fast
    ❌ No inverse_transform (can't see feature names)
    ❌ Possible hash collisions
""")

hv = HashingVectorizer(n_features=20, alternate_sign=False)
X_hash = hv.fit_transform(documents)
print(f"  Shape: {X_hash.shape}")
print(f"  Fixed at n_features={20} regardless of vocabulary size")

# Compare when to use what
print("""
  📊 When to Use What:
  ┌──────────────────┬────────────────────────────────────────┐
  │ CountVectorizer  │ Quick exploration, need feature names  │
  │ TfidfVectorizer  │ Most ML tasks (recommended default!)   │
  │ HashingVectorizer│ Very large datasets, streaming data    │
  └──────────────────┴────────────────────────────────────────┘
""")

# ── Summary ───────────────────────────────────────────────────────
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. BoW counts words; TF-IDF weights by importance
  2. TF-IDF is almost always better than plain BoW
  3. N-grams (1,2) capture word context/phrases
  4. max_df/min_df filter too common/rare words
  5. Cosine similarity measures document likeness
  6. Always start with TfidfVectorizer for ML tasks

📚 Next: 03_sentiment_analysis.py
""")
