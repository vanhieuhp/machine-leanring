"""
Sentiment Analysis - NLP Basics
================================

This module covers:
- Sentiment analysis basics
- Building sentiment classifiers
- Using TF-IDF and ML
- Using neural networks
- Real-world datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================================
# 1. SENTIMENT ANALYSIS CONCEPT
# ============================================================================

print("=" * 70)
print("1. SENTIMENT ANALYSIS CONCEPT")
print("=" * 70)

print("""
Sentiment Analysis:
- Determine the emotional tone of text
- Positive, Negative, Neutral
- Or more granular: 1-5 stars

Applications:
- Social media monitoring
- Customer reviews analysis
- Brand reputation management
- Market research
""")

# ============================================================================
# 2. SAMPLE DATASET
# ============================================================================

print("\n" + "=" * 70)
print("2. SAMPLE SENTIMENT DATA")
print("=" * 70)

# Sample movie reviews
reviews = [
    ("This movie is absolutely amazing! Best I've ever seen.", "positive"),
    ("Terrible film. Complete waste of time and money.", "negative"),
    ("It was okay. Not great, not terrible. Average.", "neutral"),
    ("I loved every minute of this movie!", "positive"),
    ("One of the worst movies I've ever watched.", "negative"),
    ("Pretty good, would recommend to friends.", "positive"),
    ("Mediocre at best. Expected much more.", "neutral"),
    ("Fantastic! A masterpiece of cinema.", "positive"),
    ("Don't waste your time. Very boring.", "negative"),
    ("It was decent. Good acting, weak plot.", "neutral"),
]

texts = [r[0] for r in reviews]
labels = [1 if r[1] == "positive" else 0 if r[1] == "negative" else 2 for r in reviews]

print(f"Sample reviews: {len(texts)}")
print("\nExamples:")
for text, label in zip(texts[:3], labels[:3]):
    sentiment = "positive" if label == 1 else "negative" if label == 0 else "neutral"
    print(f"  [{sentiment}] {text[:50]}...")

# ============================================================================
# 3. SENTIMENT WITH TF-IDF + LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("3. TF-IDF + LOGISTIC REGRESSION")
print("=" * 70)

# More reviews for better training
all_reviews = [
    ("I love this movie", 1),
    ("This is amazing", 1),
    ("Best movie ever", 1),
    ("Great film", 1),
    ("Excellent", 1),
    ("Not good", 0),
    ("Terrible", 0),
    ("Very bad", 0),
    ("Awful movie", 0),
    ("Worst film", 0),
    ("I hate it", 0),
    ("So boring", 0),
    ("Pretty good", 1),
    ("Nice one", 1),
    ("Thumbs up", 1),
    ("Not bad", 1),
    ("Could be better", 2),
    ("It's okay", 2),
    ("Average film", 2),
    ("Nothing special", 2),
]

texts = [r[0] for r in all_reviews]
labels = [r[1] for r in all_reviews]

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
y = np.array(labels)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive', 'neutral']))

# ============================================================================
# 4. SENTIMENT WITH NAIVE BAYES
# ============================================================================

print("\n" + "=" * 70)
print("4. NAIVE BAYES FOR SENTIMENT")
print("=" * 70)

print("""
Naive Bayes for Sentiment:
- MultinomialNB: Works well with text
- Assumes feature independence
- Fast and effective baseline
""")

# Train Naive Bayes
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)

y_pred_nb = nb_clf.predict(X_test)

print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")

# Compare
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ============================================================================
# 5. DEEP LEARNING FOR SENTIMENT
# ============================================================================

print("\n" + "=" * 70)
print("5. NEURAL NETWORK FOR SENTIMENT")
print("=" * 70)

if TF_AVAILABLE:
    # Prepare data
    texts = [r[0] for r in all_reviews]
    labels = [r[1] for r in all_reviews]

    # Tokenize
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(texts)

    X_seq = tokenizer.texts_to_sequences(texts)
    X_pad = pad_sequences(X_seq, maxlen=10)

    # Train/test split
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X_pad, np.array(labels), test_size=0.2, random_state=42
    )

    # Build model
    model = Sequential([
        Embedding(100, 32, input_length=10),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(X_train_nn, y_train_nn, epochs=20, verbose=0)

    # Evaluate
    _, acc = model.evaluate(X_test_nn, y_test_nn, verbose=0)
    print(f"Neural Network Accuracy: {acc:.4f}")

# ============================================================================
# 6. REAL-WORLD DATASET: IMDB
# ============================================================================

print("\n" + "=" * 70)
print("6. IMDB DATASET")
print("=" * 70)

if TF_AVAILABLE:
    print("""
IMDB Movie Review Dataset:
- 50,000 reviews (25k train, 25k test)
- Binary sentiment (positive/negative)
- Standard benchmark for sentiment analysis
""")

    # Load IMDB
    (X_train_imdb, y_train_imdb), (X_test_imdb, y_test_imdb) = keras.datasets.imdb.load_data(num_words=10000)

    print(f"Train samples: {len(X_train_imdb)}")
    print(f"Test samples: {len(X_test_imdb)}")

    # Pad sequences
    maxlen = 200
    X_train_pad = keras.preprocessing.sequence.pad_sequences(X_train_imdb, maxlen=maxlen)
    X_test_pad = keras.preprocessing.sequence.pad_sequences(X_test_imdb, maxlen=maxlen)

    # Simple model
    model_imdb = Sequential([
        Embedding(10000, 64, input_length=maxlen),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model_imdb.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train (small subset for demo)
    model_imdb.fit(
        X_train_pad[:5000], y_train_imdb[:5000],
        epochs=3,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    loss, acc = model_imdb.evaluate(X_test_pad[:1000], y_test_imdb[:1000], verbose=0)
    print(f"\nIMDB Test Accuracy: {acc:.4f}")

# ============================================================================
# 7. HANDLING REAL REVIEWS
# ============================================================================

print("\n" + "=" * 70)
print("7. PRACTICAL SENTIMENT PIPELINE")
print("=" * 70)

import re

def preprocess_for_sentiment(text):
    """Clean text for sentiment analysis"""
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Keep important punctuation
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)

    return text

def predict_sentiment(text, vectorizer, model):
    """Predict sentiment of a review"""
    # Preprocess
    text_clean = preprocess_for_sentiment(text)

    # Vectorize
    X = vectorizer.transform([text_clean])

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return pred, proba

# Test
test_reviews = [
    "This movie is absolutely fantastic! Best ever!",
    "Terrible movie. Complete waste of time.",
    "It was okay, nothing special."
]

print("Testing sentiment predictions:")
print("-" * 50)

for review in test_reviews:
    pred, proba = predict_sentiment(review, tfidf, clf)
    sentiment = "positive" if pred == 1 else "negative" if pred == 0 else "neutral"
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print(f"Probability: {proba}")
    print()

# ============================================================================
# 8. COMPARING MODELS
# ============================================================================

print("\n" + "=" * 70)
print("8. MODEL COMPARISON")
print("=" * 70)

print("""
Sentiment Analysis Approaches:

1. Rule-based:
   - Lexicons (positive/negative word lists)
   - Fast, interpretable
   - Limited accuracy

2. Traditional ML:
   - TF-IDF + Logistic Regression
   - TF-IDF + Naive Bayes
   - Fast, good baseline

3. Deep Learning:
   - LSTM/BERT
   - Best accuracy
   - Requires more data/compute

4. Pre-trained Models:
   - VADER, TextBlob
   - Ready to use
   - Good for quick results
""")

# ============================================================================
# 9. VADER SENTIMENT
# ============================================================================

print("\n" + "=" * 70)
print("9. VADER SENTIMENT")
print("=" * 70)

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)

    sia = SentimentIntensityAnalyzer()

    test_text = "This is a really amazing and wonderful movie!"

    scores = sia.polarity_scores(test_text)

    print(f"Text: {test_text}")
    print(f"Scores: {scores}")

    if scores['compound'] >= 0.05:
        sentiment = "positive"
    elif scores['compound'] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    print(f"Sentiment: {sentiment}")

except ImportError:
    print("NLTK VADER not available")

# ============================================================================
# 10. BEST PRACTICES
# ============================================================================

print("\n" + "=" * 70)
print("10. BEST PRACTICES")
print("=" * 70)

print("""
Sentiment Analysis Best Practices:

1. Preprocessing:
   - Handle negations ("not good")
   - Keep emoticons/emojis
   - Handle slang

2. Feature Engineering:
   - N-grams (bigrams capture "not good")
   - Character-level for typos
   - Combine with embeddings

3. Model Selection:
   - Start with TF-IDF + Logistic Regression
   - Try neural networks for better accuracy
   - Fine-tune BERT for best results

4. Data Quality:
   - Balanced classes
   - Clean labels
   - Enough data
""")

print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. Sentiment: Positive/Negative/Neutral
2. TF-IDF + ML: Good baseline
3. Deep learning: Best accuracy
4. Pre-trained tools: Quick results
5. Preprocessing matters for negations

Common Datasets:
- IMDB Movie Reviews
- Amazon Reviews
- Twitter Sentiment
- YELP Reviews

Next Steps:
- Build complete sentiment pipeline
- Try BERT for state-of-the-art
- Handle multi-class sentiment
""")
