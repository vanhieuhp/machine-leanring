# 📝 NLP Basics — Deep Dive Guide

## 📋 Table of Contents

1. [Overview & Intuition](#overview--intuition)
2. [Learning Roadmap](#learning-roadmap)
3. [Text Preprocessing](#1-text-preprocessing)
4. [Text Representation](#2-text-representation)
5. [Sentiment Analysis](#3-sentiment-analysis)
6. [Text Classification](#4-text-classification)
7. [Word Embeddings](#5-word-embeddings)
8. [Key Takeaways](#key-takeaways)

---

## Overview & Intuition

**Core Idea**: Transform human language (unstructured text) into numerical representations that machines can understand and learn from.

### The NLP Pipeline

```
Raw Text → Preprocessing → Representation → Model → Prediction
  "I love this!"                                      "Positive 😊"

Detailed pipeline:
  1. Clean text (lowercase, remove punctuation)
  2. Tokenize (split into words/tokens)
  3. Remove stopwords ("the", "is", "a")
  4. Stem/Lemmatize ("running" → "run")
  5. Vectorize (text → numbers)
  6. Train model on numerical features
  7. Predict on new text
```

### NLP Task Categories

| Task | Example | Approach |
|------|---------|----------|
| **Text Classification** | Spam detection | BoW/TF-IDF + ML classifier |
| **Sentiment Analysis** | Review rating | TF-IDF + LogReg/NB |
| **Named Entity Recognition** | Extract names, places | spaCy / BERT |
| **Machine Translation** | English → French | Seq2Seq / Transformer |
| **Text Summarization** | Long article → summary | Extractive / Abstractive |
| **Question Answering** | Q&A system | Transformer (BERT) |

---

## Learning Roadmap

| Day | Topic | Study File | Time |
|-----|-------|-----------|------|
| 1 | Text preprocessing & cleaning | `01_text_preprocessing.py` | 2-3h |
| 2 | BoW, TF-IDF, vectorization | `02_text_representation.py` | 2-3h |
| 3 | Sentiment analysis project | `03_sentiment_analysis.py` | 3h |
| 4 | Text classification pipeline | `04_text_classification.py` | 3h |
| 5 | Exercises | `exercises.py` | 3h |

---

## 1. Text Preprocessing

### Step-by-Step Pipeline

```python
# Raw text
"I can't BELIEVE it!!! This is the BEST movie I've ever seen 🎬"

# Step 1: Lowercase
"i can't believe it!!! this is the best movie i've ever seen 🎬"

# Step 2: Remove special characters
"i cant believe it this is the best movie ive ever seen"

# Step 3: Tokenize
["i", "cant", "believe", "it", "this", "is", "the", "best",
 "movie", "ive", "ever", "seen"]

# Step 4: Remove stopwords
["cant", "believe", "best", "movie", "ever", "seen"]

# Step 5: Lemmatize
["cant", "believe", "best", "movie", "ever", "see"]
```

### Stemming vs. Lemmatization

| Stemming | Lemmatization |
|----------|---------------|
| Chops word endings | Uses dictionary lookup |
| "running" → "run" | "running" → "run" |
| "better" → "better" | "better" → "good" |
| "flies" → "fli" ❌ | "flies" → "fly" ✅ |
| Fast but crude | Slower but accurate |
| Porter Stemmer | WordNet Lemmatizer |

---

## 2. Text Representation

### Bag of Words (BoW)

```
Vocabulary: ["amazing", "bad", "good", "movie", "terrible"]

"This movie is good"      → [0, 0, 1, 1, 0]
"This movie is bad"       → [0, 1, 0, 1, 0]
"Amazing movie"            → [1, 0, 0, 1, 0]

Properties:
  ✅ Simple, interpretable
  ❌ Ignores word order ("good not bad" ≈ "bad not good")
  ❌ Large, sparse matrices
  ❌ All words weighted equally
```

### TF-IDF (Term Frequency × Inverse Document Frequency)

```
TF(word, doc) = count(word in doc) / total words in doc
IDF(word) = log(total docs / docs containing word)
TF-IDF = TF × IDF

Effect: common words (high DF) get LOW weight
        unique words (low DF) get HIGH weight

"the" appears everywhere → IDF ≈ 0 → TF-IDF ≈ 0 (ignored!)
"quantum" appears rarely → IDF high → TF-IDF high (important!)
```

### Word Embeddings

```
Traditional:  "king" → [0, 0, 0, 1, 0, 0, ...]  (one-hot, sparse)
Embeddings:   "king" → [0.23, -0.45, 0.78, ...]  (dense, 100-300 dims)

Magic property:
  king - man + woman ≈ queen
  paris - france + italy ≈ rome

Common embeddings:
  Word2Vec:  Google, 2013.  CBOW or Skip-gram
  GloVe:     Stanford, 2014. Global co-occurrence matrix
  FastText:  Facebook, 2016. Subword embeddings
```

### Comparison

| Method | Dimensionality | Word Order | Semantics | Speed |
|--------|---------------|------------|-----------|-------|
| BoW | Very high (vocab size) | ❌ | ❌ | Fast |
| TF-IDF | Very high (vocab size) | ❌ | Partial | Fast |
| Word2Vec | Low (100-300) | ❌ | ✅ | Medium |
| BERT | Medium (768) | ✅ | ✅✅ | Slow |

---

## 3. Sentiment Analysis

### Approach 1: Rule-based
```
Positive words: {"good", "great", "amazing", "love", "excellent"}
Negative words: {"bad", "terrible", "awful", "hate", "poor"}

Count positive - negative → sentiment score
Simple but limited!
```

### Approach 2: ML-based (recommended)
```
1. Collect labeled data (text + sentiment label)
2. Preprocess text
3. Vectorize with TF-IDF
4. Train classifier (Naive Bayes, LogReg, SVM)
5. Predict on new text
```

### Approach 3: Deep Learning
```
1. Tokenize and encode text
2. Use embedding layer (pre-trained or learned)
3. Feed into LSTM/CNN/Transformer
4. Output sentiment probability
```

---

## 4. Text Classification

### General Pipeline

```
Raw Text
  → Preprocessing (clean, tokenize, etc.)
  → Feature Extraction (TF-IDF or embeddings)
  → Model Training (Naive Bayes, LogReg, SVM, BERT)
  → Evaluation (accuracy, F1, confusion matrix)
  → Prediction

Best starter models for text classification:
  1. MultinomialNB + TF-IDF  (fast baseline)
  2. LogisticRegression + TF-IDF (strong baseline)
  3. LinearSVC + TF-IDF (best traditional ML)
  4. BERT fine-tuning (state-of-the-art, needs GPU)
```

---

## 5. Word Embeddings

### Word2Vec Architectures

```
CBOW (Continuous Bag of Words):
  Context words → predict center word
  "The cat ___ on the mat" → predict "sat"
  Faster training, better for frequent words

Skip-gram:
  Center word → predict context words
  "sat" → predict ["The", "cat", "on", "the", "mat"]
  Better for rare words, slower
```

### Using Pre-trained Embeddings

```python
# Option 1: Gensim Word2Vec (train your own)
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)

# Option 2: Load pre-trained GloVe
# Download from: nlp.stanford.edu/projects/glove/
# 6B tokens, 400K vocab, 50/100/200/300 dimensions

# Option 3: spaCy (built-in word vectors)
import spacy
nlp = spacy.load("en_core_web_md")  # includes 300d vectors
doc = nlp("king")
print(doc.vector.shape)  # (300,)
```

---

## Key Takeaways

1. **Always preprocess** — clean text before any analysis
2. **Start with TF-IDF + LogReg** — surprisingly strong baseline
3. **Naive Bayes** is fast and works well for text classification
4. **TF-IDF > BoW** in almost all cases (importance weighting)
5. **Word embeddings** capture semantic meaning
6. **For production NLP** in 2024+, Transformers (BERT) dominate

### Decision Flowchart

```
Simple text classification?
  └── YES → TF-IDF + LogisticRegression (fast, interpretable)

Need semantic understanding?
  └── YES → Pre-trained embeddings + LSTM/CNN

State-of-the-art accuracy needed?
  └── YES → Fine-tune BERT/RoBERTa (needs GPU)

Real-time / low resource?
  └── YES → TF-IDF + LinearSVC or distilled BERT
```

---

## Study Files

| # | File | Description | Difficulty |
|---|------|-------------|------------|
| 1 | `01_text_preprocessing.py` | Cleaning, tokenization, stemming, lemmatization | ⭐⭐ |
| 2 | `02_text_representation.py` | BoW, TF-IDF, n-grams, hashing | ⭐⭐ |
| 3 | `03_sentiment_analysis.py` | Sentiment analysis with ML | ⭐⭐⭐ |
| 4 | `04_text_classification.py` | Multi-class text classification | ⭐⭐⭐ |
| 5 | `exercises.py` | 5 practice problems | ⭐⭐⭐ |
