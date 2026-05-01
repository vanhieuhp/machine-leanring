"""
NLP Basics - Exercises
=======================

Practice problems for NLP.
"""

import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Try to import NLTK
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ============================================================================
# EXERCISE 1: Text Preprocessing
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Text Preprocessing")
print("=" * 70)

print("""
Task:
1. Create a function to clean text
2. Handle: lowercase, URLs, special characters
3. Apply to sample reviews
""")

reviews = [
    "This is AMAZING!!! BEST product ever http://example.com",
    "TERRIBLE! <b>Do not buy</b> this is a scam!!!",
    "It's okay. Not great but works fine."
]

def clean_text(text):
    # Your code here:
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

print("Original vs Cleaned:")
for review in reviews:
    cleaned = clean_text(review)
    print(f"  Original:  {review}")
    print(f"  Cleaned:  {cleaned}")
    print()

# ============================================================================
# EXERCISE 2: Tokenization
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Tokenization")
print("=" * 70)

print("""
Task:
1. Tokenize sample sentences
2. Count tokens
3. Compare approaches
""")

text = "I love natural language processing! It's fascinating."

if NLTK_AVAILABLE:
    # NLTK tokenization
    tokens = word_tokenize(text)
    print(f"NLTK tokens: {tokens}")
else:
    # Simple split
    tokens = text.split()
    print(f"Simple tokens: {tokens}")

# Count
word_count = len(tokens)
unique_count = len(set(tokens))
print(f"Word count: {word_count}")
print(f"Unique words: {unique_count}")

# ============================================================================
# EXERCISE 3: Stopword Removal
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Stopword Removal")
print("=" * 70)

print("""
Task:
1. Remove stopwords from sentences
2. Compare with/without stopwords
""")

sentence = "The quick brown fox jumps over the lazy dog"

words = sentence.lower().split()

if NLTK_AVAILABLE:
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
else:
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

filtered = [w for w in words if w not in stop_words]

print(f"Original: {words}")
print(f"After removing stopwords: {filtered}")

# ============================================================================
# EXERCISE 4: TF-IDF Features
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: TF-IDF Features")
print("=" * 70)

print("""
Task:
1. Create TF-IDF vectors
2. Compare different documents
3. Find important words
""")

documents = [
    "The cat sat on the mat",
    "The dog played in the park",
    "Cats and dogs are pets"
]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

print(f"Vocabulary: {tfidf.get_feature_names_out()}")
print(f"\nTF-IDF Matrix:")
import pandas as pd
df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
print(df.round(3))

# ============================================================================
# EXERCISE 5: Sentiment Classification
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Sentiment Classification")
print("=" * 70)

print("""
Task:
1. Create training data
2. Train classifier
3. Predict sentiment
""")

data = [
    ("I love this", 1),
    ("Great product", 1),
    ("Amazing quality", 1),
    ("Very bad", 0),
    ("Terrible", 0),
    ("Worst ever", 0),
]

texts = [d[0] for d in data]
labels = [d[1] for d in data]

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
y = labels

# Train
clf = LogisticRegression()
clf.fit(X, y)

# Test
test = ["This is awesome", "Not good at all"]
X_test = tfidf.transform(test)
predictions = clf.predict(X_test)

print("Sentiment Predictions:")
for text, pred in zip(test, predictions):
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"  '{text}' → {sentiment}")

# ============================================================================
# EXERCISE 6: Text Classification
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Text Classification")
print("=" * 70)

print("""
Task:
1. Create topic classification data
2. Train multi-class classifier
3. Evaluate
""")

data = [
    ("Apple releases new iPhone", "tech"),
    ("Tech stock prices rise", "tech"),
    ("Football team wins championship", "sports"),
    ("Athlete breaks world record", "sports"),
    ("Bank reports record profits", "finance"),
    ("Stock market analysis", "finance"),
]

texts = [d[0] for d in data]
labels = [d[1] for d in data]

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)

# Train
clf = LogisticRegression()
clf.fit(X, labels)

# Test
test_texts = [
    "New smartphone technology",
    "Finance minister announces policy",
    "Basketball game highlights"
]
X_test = tfidf.transform(test_texts)
predictions = clf.predict(X_test)

print("Topic Predictions:")
for text, pred in zip(test_texts, predictions):
    print(f"  '{text}' → {pred}")

# ============================================================================
# BONUS: Regular Expressions
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: Regular Expressions")
print("=" * 70)

print("""
Task:
1. Extract emails from text
2. Extract phone numbers
3. Extract hashtags
""")

text = "Contact us at info@company.com or call 555-123-4567. Follow us @company! #sales #marketing"

# Extract emails
emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
print(f"Emails: {emails}")

# Extract phones
phones = re.findall(r'\d{3}-\d{3}-\d{4}', text)
print(f"Phones: {phones}")

# Extract hashtags
hashtags = re.findall(r'#\w+', text)
print(f"Hashtags: {hashtags}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
What you practiced:
1. Text cleaning and preprocessing
2. Tokenization
3. Stopword removal
4. TF-IDF feature extraction
5. Sentiment classification
6. Topic classification
7. Regular expressions

Key Takeaways:
1. Preprocessing is crucial for NLP
2. TF-IDF is a strong baseline
3. Simple ML works well for text
4. Regular expressions for pattern matching

Next Steps:
- Try deep learning for NLP
- Explore word embeddings
- Build complete NLP pipelines
""")
