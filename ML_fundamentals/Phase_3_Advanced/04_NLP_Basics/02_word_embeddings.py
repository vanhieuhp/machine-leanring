"""
Word Embeddings - NLP Basics
============================

This module covers:
- Bag of Words
- TF-IDF
- Word embeddings
- Word2Vec, GloVe
- Using embeddings in neural networks
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 1. BAG OF WORDS
# ============================================================================

print("=" * 70)
print("1. BAG OF WORDS (BoW)")
print("=" * 70)

print("""
Bag of Words: Simple text representation
- Count word occurrences
- Ignore word order

Steps:
1. Create vocabulary (unique words)
2. Count occurrences in each document

Limitations:
- Ignores word order
- Can't handle new words easily
- Sparse representation
""")

# Example documents
documents = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "I love my cat and my dog"
]

# Create BoW vectorizer
bow = CountVectorizer()
bow_matrix = bow.fit_transform(documents)

print(f"Vocabulary: {bow.get_feature_names_out()}")
print(f"\nBoW Matrix:\n{bow_matrix.toarray()}")

# Show as DataFrame
import pandas as pd
df_bow = pd.DataFrame(bow_matrix.toarray(), columns=bow.get_feature_names_out())
print(f"\nDataFrame view:")
print(df_bow)

# ============================================================================
# 2. N-GRAMS
# ============================================================================

print("\n" + "=" * 70)
print("2. N-GRAMS")
print("=" * 70)

print("""
N-grams: Capture word sequences

- Unigrams: single words (default)
- Bigrams: 2-word sequences
- Trigrams: 3-word sequences
- ...

Why use n-grams:
- Capture word order
- Preserve context
- "not good" vs "good"
""")

# N-gram example
sentence = "The cat sat on the mat"

# Unigrams
unigram = CountVectorizer(ngram_range=(1, 1))
unigrams = unigram.fit_transform([sentence])
print(f"Unigrams: {unigram.get_feature_names_out()}")

# Bigrams
bigram = CountVectorizer(ngram_range=(2, 2))
bigrams = bigram.fit_transform([sentence])
print(f"Bigrams: {bigram.get_feature_names_out()}")

# Trigrams
trigram = CountVectorizer(ngram_range=(3, 3))
trigrams = trigram.fit_transform([sentence])
print(f"Trigrams: {trigram.get_feature_names_out()}")

# ============================================================================
# 3. TF-IDF
# ============================================================================

print("\n" + "=" * 70)
print("3. TF-IDF")
print("=" * 70)

print("""
TF-IDF: Term Frequency - Inverse Document Frequency

TF (Term Frequency): How often a word appears in document
- tf(t,d) = count of t in d

IDF (Inverse Document Frequency): How unique is the word
- idf(t) = log(N / df(t))
- N = total documents
- df(t) = documents containing t

TF-IDF = TF × IDF

Why it works:
- Common words get lower scores
- Rare, distinctive words get higher scores
""")

# Example documents
docs = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "I love my cat and my dog",
    "The cat and dog are friends"
]

# TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)

print(f"Vocabulary: {tfidf.get_feature_names_out()}")
print(f"\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Show as DataFrame
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)
print(f"\nDataFrame view:")
print(df_tfidf.round(3))

# ============================================================================
# 4. COSINE SIMILARITY
# ============================================================================

print("\n" + "=" * 70)
print("4. COSINE SIMILARITY")
print("=" * 70)

print("""
Cosine Similarity: Measure similarity between documents
- Range: -1 to 1
- 1 = identical
- 0 = orthogonal (no similarity)
- -1 = opposite

Formula:
cosine(A, B) = (A · B) / (||A|| × ||B||)
""")

doc1 = "The cat sat on the mat"
doc2 = "A dog ran in the park"
doc3 = "The cat loves the dog"

# Vectorize
tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform([doc1, doc2, doc3])

# Calculate similarities
sim_12 = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
sim_13 = cosine_similarity(matrix[0:1], matrix[2:3])[0][0]
sim_23 = cosine_similarity(matrix[1:2], matrix[2:3])[0][0]

print(f"Similarity between:")
print(f"  Doc1 & Doc2: {sim_12:.4f}")
print(f"  Doc1 & Doc3: {sim_13:.4f}")
print(f"  Doc2 & Doc3: {sim_23:.4f}")

# ============================================================================
# 5. WORD EMBEDDINGS
# ============================================================================

print("\n" + "=" * 70)
print("5. WORD EMBEDDINGS")
print("=" * 70)

print("""
Word Embeddings: Dense vector representations

Why embeddings?
- BoW/TF-IDF are sparse (huge, mostly zeros)
- Embeddings are dense (small, meaningful)
- Capture semantic relationships
- Similar words → similar vectors

Types:
1. Word2Vec: Predict context from word (CBOW) or word from context (Skip-gram)
2. GloVe: Global Vectors (matrix factorization)
3. FastText: Subword embeddings

Properties:
- king - man + woman ≈ queen
- Paris - France + Japan ≈ Tokyo
""")

# ============================================================================
# 6. USING WORD2VEC (GENSIM)
# ============================================================================

print("\n" + "=" * 70)
print("6. WORD2VEC")
print("=" * 70)

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim not installed. Install with: pip install gensim")

if GENSIM_AVAILABLE:
    # Sample corpus
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'ran', 'in', 'the', 'park'],
        ['cats', 'and', 'dogs', 'are', 'friends'],
        ['the', 'cat', 'loves', 'the', 'dog'],
        ['dogs', 'love', 'to', 'play', 'in', 'the', 'park']
    ]

    # Train Word2Vec
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    print("Word2Vec trained!")

    # Get word vectors
    vector_cat = model.wv['cat']
    vector_dog = model.wv['dog']
    print(f"\nWord vector dimensions: {len(vector_cat)}")

    # Find similar words
    print("\nSimilar words to 'cat':")
    similar = model.wv.most_similar('cat', topn=3)
    for word, score in similar:
        print(f"  {word}: {score:.4f}")

    # Similarity
    sim = model.wv.similarity('cat', 'dog')
    print(f"\nSimilarity between 'cat' and 'dog': {sim:.4f}")
else:
    print("Gensim not available. Skipping Word2Vec demo.")

# ============================================================================
# 7. PRE-TRAINED EMBEDDINGS
# ============================================================================

print("\n" + "=" * 70)
print("7. PRE-TRAINED EMBEDDINGS")
print("=" * 70)

print("""
Pre-trained Embeddings:
- Trained on large corpora
- Download and use directly
- GloVe, FastText, Word2Vec

Where to get:
- GloVe: https://nlp.stanford.edu/projects/glove
- FastText: https://fasttext.cc
- TensorFlow Hub

Using in Keras:
- Embedding layer
- Load pre-trained weights
""")

# ============================================================================
# 8. KERAS EMBEDDING LAYER
# ============================================================================

print("\n" + "=" * 70)
print("8. KERAS EMBEDDING LAYER")
print("=" * 70)

print("""
Keras Embedding Layer:
- Maps word indices to dense vectors
- Can be trained or use pre-trained

Parameters:
- input_dim: vocabulary size
- output_dim: embedding dimension
- input_length: sequence length
""")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Dense, Flatten
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

if TF_AVAILABLE:
    # Simple example
    vocab_size = 10000
    embedding_dim = 100
    max_length = 50

    # Create embedding layer
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length
    )

    # Example input (batch of sequences)
    example_input = tf.constant([[1, 5, 10, 23, 45], [2, 8, 12, 3, 99]])

    # Get embeddings
    embeddings = embedding_layer(example_input)
    print(f"Input shape: {example_input.shape}")
    print(f"Embedding output shape: {embeddings.shape}")
    print(f"  (batch_size, sequence_length, embedding_dim)")
else:
    print("TensorFlow not available")

# ============================================================================
# 9. PRACTICAL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL COMPARISON")
print("=" * 70)

print("""
| Method       | Dimensions | Use Case                    |
|--------------|------------|-----------------------------|
| Bag of Words | High/Sparse| Simple, short documents    |
| TF-IDF       | High/Sparse| Feature weighting           |
| Word2Vec     | Medium/Dense| Semantic similarity        |
| GloVe        | Medium/Dense| Pre-trained, large corpus |
| FastText     | Medium/Dense| OOV handling, subwords    |

When to use what:
- Quick baseline: BoW or TF-IDF
- Semantic similarity: Word2Vec or GloVe
- Deep learning: Embedding layer
- Unknown words: FastText
""")

# ============================================================================
# 10. TEXT CLASSIFICATION WITH TF-IDF
# ============================================================================

print("\n" + "=" * 70)
print("10. TEXT CLASSIFICATION WITH TF-IDF")
print("=" * 70)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

try:
    # Load small subset for demo
    categories = ['rec.sport.baseball', 'sci.space']
    newsgroups = fetch_20newsgroups(categories=categories)

    X = newsgroups.data[:500]  # Limit for speed
    y = newsgroups.target[:500]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print(f"TF-IDF + Logistic Regression Accuracy: {acc:.4f}")
except Exception as e:
    print(f"Could not load newsgroups: {e}")
    print("Using simple example instead...")

    # Simple example
    X = ["I love this product", "This is terrible", "Amazing quality", "Worst purchase", "Great item"]
    y = [1, 0, 1, 0, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    print(f"Simple Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n" + "=" * 70)
print("WORD EMBEDDINGS SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. BoW: Simple word counting
2. TF-IDF: Weighted word importance
3. Cosine similarity: Document similarity
4. Word embeddings: Dense semantic vectors
5. Word2Vec: Learn embeddings from corpus
6. Pre-trained embeddings: Use existing knowledge

Next Steps:
- Learn sentiment analysis
- Build text classifiers
- Explore deep learning for NLP
""")
