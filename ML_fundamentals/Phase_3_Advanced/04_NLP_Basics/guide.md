# NLP Basics Guide

## What is NLP?

Natural Language Processing enables computers to understand and process human language.

## Key Tasks

### 1. Text Preprocessing
- Tokenization: Split into words
- Lowercasing: Normalize case
- Removing stopwords: Remove common words
- Stemming/Lemmatization: Reduce to root form

### 2. Text Representation
- Bag of Words: Count word frequencies
- TF-IDF: Weight by importance
- Word Embeddings: Dense vectors (Word2Vec, GloVe)

### 3. Text Classification
- Sentiment analysis
- Spam detection
- Topic classification

### 4. Named Entity Recognition
- Identify people, places, organizations
- Extract information

### 5. Machine Translation
- Translate between languages
- Sequence-to-sequence models

## Common Techniques

### Tokenization
```python
text = "Hello, world!"
tokens = text.split()  # ["Hello,", "world!"]
```

### Stopword Removal
```python
stopwords = {"the", "a", "is", "and"}
filtered = [w for w in tokens if w not in stopwords]
```

### TF-IDF
- TF: How often word appears in document
- IDF: How unique word is across documents
- TF-IDF = TF * IDF

### Word Embeddings
- Dense vectors representing words
- Similar words have similar vectors
- Learned from large text corpora

## Tools

- **NLTK**: Natural Language Toolkit
- **spaCy**: Industrial NLP
- **TextBlob**: Simple text processing
- **Transformers**: State-of-the-art models

## Applications

- Sentiment analysis
- Chatbots
- Machine translation
- Question answering
- Text summarization

## Study Files

1. `01_text_preprocessing.py` - Clean and prepare text
2. `02_text_representation.py` - Vectorize text
3. `03_sentiment_analysis.py` - Analyze sentiment
4. `04_text_classification.py` - Classify documents
5. `exercises.py` - Practice problems
