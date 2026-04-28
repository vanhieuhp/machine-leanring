"""
Text Classification - NLP Basics
================================

This module covers:
- Text classification basics
- Multi-class classification
- Building classifiers with neural networks
- Using pre-trained models
- Real-world applications
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Embedding, Flatten, Dropout,
        Conv1D, GlobalMaxPooling1D
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================================
# 1. TEXT CLASSIFICATION OVERVIEW
# ============================================================================

print("=" * 70)
print("1. TEXT CLASSIFICATION OVERVIEW")
print("=" * 70)

print("""
Text Classification:
- Assign predefined categories to text
- Binary: yes/no classification
- Multi-class: multiple categories
- Multi-label: multiple labels per text

Common Applications:
- Spam detection
- Topic classification
- Sentiment analysis
- Intent detection
- Language identification
""")

# ============================================================================
# 2. SIMPLE TEXT CLASSIFIER
# ============================================================================

print("\n" + "=" * 70)
print("2. SIMPLE TEXT CLASSIFIER")
print("=" * 70)

# Sample dataset: Topic classification
documents = [
    ("The stock market dropped significantly today", "finance"),
    ("Scientists discovered a new planet", "science"),
    ("The football team won the championship", "sports"),
    ("Apple released new iPhone features", "technology"),
    ("The team scored a goal in the final minute", "sports"),
    ("New COVID-19 vaccine shows promise", "health"),
    ("The government announced new tax policies", "finance"),
    ("Mars rover sends back new images", "science"),
    ("Tech companies report record earnings", "technology"),
    ("Exercise improves heart health", "health"),
]

texts = [d[0] for d in documents]
labels = [d[1] for d in documents]

# Convert labels to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(labels)

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClasses: {le.classes_}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================================================
# 3. MULTI-CLASS CLASSIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("3. MULTI-CLASS CLASSIFICATION")
print("=" * 70)

# Generate more data for each class
texts_multi = []
labels_multi = []

categories = {
    'tech': ['New smartphone released', 'AI technology advances', 'Software update available',
             'Tech giant announces new product', 'Computer processor speeds up'],
    'sports': ['Team wins championship', 'Player scores goal', 'Match ends in tie',
               'Athlete breaks record', 'Game postponed due to rain'],
    'finance': ['Stock market rises', 'Bank announces profits', 'Economy shows growth',
                'Interest rates increase', 'Company reports earnings'],
    'health': ['Doctor recommends exercise', 'New medicine approved', 'Healthy diet benefits',
               'Hospital expands capacity', 'Wellness trend grows'],
}

for category, examples in categories.items():
    for text in examples:
        texts_multi.append(text)
        labels_multi.append(category)

# Vectorize
tfidf = TfidfVectorizer()
X_multi = tfidf.fit_transform(texts_multi)

# Train
clf_multi = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf_multi.fit(X_multi, labels_multi)

# Test new examples
test_texts = [
    "New smartphone features amazing camera",
    "Stock prices fall sharply",
    "Star athlete wins gold medal"
]

X_test_new = tfidf.transform(test_texts)
predictions = clf_multi.predict(X_test_new)

print("Predictions:")
for text, pred in zip(test_texts, predictions):
    print(f"  '{text}' → {pred}")

# ============================================================================
# 4. NEURAL NETWORK CLASSIFIER
# ============================================================================

print("\n" + "=" * 70)
print("4. NEURAL NETWORK CLASSIFIER")
print("=" * 70)

if TF_AVAILABLE:
    # Prepare data
    texts = texts_multi
    labels = labels_multi

    # Tokenize
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(texts)

    X_seq = tokenizer.texts_to_sequences(texts)
    X_pad = pad_sequences(X_seq, maxlen=10)

    # Encode labels
    le_nn = LabelEncoder()
    y_nn = le_nn.fit_transform(labels)

    # Train/test
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        X_pad, y_nn, test_size=0.2, random_state=42
    )

    # Build model
    model = Sequential([
        Embedding(100, 32, input_length=10),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(X_train_nn, y_train_nn, epochs=30, verbose=0)

    # Evaluate
    _, acc = model.evaluate(X_test_nn, y_test_nn, verbose=0)
    print(f"Neural Network Accuracy: {acc:.4f}")

# ============================================================================
# 5. CNN FOR TEXT
# ============================================================================

print("\n" + "=" * 70)
print("5. CNN FOR TEXT CLASSIFICATION")
print("=" * 70)

print("""
Text CNN:
- Apply 1D convolutions over word embeddings
- Capture local patterns
- Fast and effective
""")

if TF_AVAILABLE:
    # CNN model
    cnn_model = Sequential([
        Embedding(100, 64, input_length=10),
        Conv1D(128, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])

    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    cnn_model.fit(X_train_nn, y_train_nn, epochs=20, verbose=0)

    _, acc = cnn_model.evaluate(X_test_nn, y_test_nn, verbose=0)
    print(f"CNN Accuracy: {acc:.4f}")

# ============================================================================
# 6. SPAM DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("6. SPAM DETECTION")
print("=" * 70)

# Sample spam data
spam_data = [
    ("Congratulations! You won a free iPhone!", "spam"),
    ("Click here to claim your prize", "spam"),
    ("Urgent: Your account needs verification", "spam"),
    ("Buy now! Limited time offer", "spam"),
    ("You have been selected for a special offer", "spam"),
    ("Hi, how are you doing?", "ham"),
    ("Can we meet tomorrow?", "ham"),
    ("The meeting is at 3pm", "ham"),
    ("Thanks for your help", "ham"),
    ("What time is the party?", "ham"),
]

spam_texts = [d[0] for d in spam_data]
spam_labels = [1 if d[1] == "spam" for d in spam_data]

# Vectorize
tfidf_spam = TfidfVectorizer()
X_spam = tfidf_spam.fit_transform(spam_texts)

# Train
clf_spam = LogisticRegression(max_iter=1000)
clf_spam.fit(X_spam, spam_labels)

# Test
test_spam = [
    "Free money! Click now!",
    "Are you coming to the meeting?",
    "You have won a lottery!",
    "Let's have lunch"
]

X_test_spam = tfidf_spam.transform(test_spam)
predictions = clf_spam.predict(X_test_spam)

print("Spam Detection:")
for text, pred in zip(test_spam, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"  '{text}' → {label}")

# ============================================================================
# 7. INTENT CLASSIFICATION
# ============================================================================

print("\n=" * 70)
print("7. INTENT CLASSIFICATION")
print("=" * 70)

print("""
Intent Classification:
- Classify user message into intents
- Used in chatbots
- Example intents: greeting, order, cancel, track
""")

# Chatbot intents
intents = {
    'greeting': ['Hello', 'Hi there', 'Good morning', 'Hey'],
    'order': ['I want to order', 'Can I buy this', 'Add to cart'],
    'cancel': ['Cancel my order', 'I want to cancel', 'Remove this'],
    'track': ['Where is my order', 'Track shipment', 'Delivery status'],
}

texts_intent = []
labels_intent = []

for intent, examples in intents.items():
    for text in examples:
        texts_intent.append(text)
        labels_intent.append(intent)

# Vectorize
tfidf_intent = TfidfVectorizer()
X_intent = tfidf_intent.fit_transform(texts_intent)

clf_intent = LogisticRegression(max_iter=1000)
clf_intent.fit(X_intent, labels_intent)

# Test
test_intents = [
    "I need to buy something",
    "Hello! How are you?",
    "Where is my package?",
    "Cancel order please"
]

X_test_intent = tfidf_intent.transform(test_intents)
predictions = clf_intent.predict(X_test_intent)

print("Intent Classification:")
for text, pred in zip(test_intents, predictions):
    print(f"  '{text}' → {pred}")

# ============================================================================
# 8. NEWS CATEGORIZATION
# ============================================================================

print("\n" + "=" * 70)
print("8. NEWS CATEGORIZATION")
print("=" * 70)

print("""
News Categorization:
- Classify news articles into categories
- Categories: Politics, Sports, Business, Tech, etc.
- Similar to topic classification
""")

# ============================================================================
# 9. BEST PRACTICES
# ============================================================================

print("\n" + "=" * 70)
print("9. BEST PRACTICES")
print("=" * 70)

print("""
Text Classification Best Practices:

1. Data Quality:
   - Clean, labeled data
   - Balanced classes
   - Enough samples per class

2. Feature Engineering:
   - TF-IDF: Good baseline
   - N-grams: Capture phrases
   - Word embeddings: Semantic understanding

3. Model Selection:
   - Start: Logistic Regression
   - Medium: CNN/LSTM
   - Best: BERT/Transformers

4. Evaluation:
   - Use proper metrics
   - Check confusion matrix
   - Handle class imbalance
""")

# ============================================================================
# 10. PIPELINE SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("10. COMPLETE PIPELINE")
print("=" * 70)

print("""
Text Classification Pipeline:

1. Data Collection
   - Gather labeled data
   - Ensure quality

2. Preprocessing
   - Clean text
   - Tokenize
   - Remove stopwords (optional)

3. Feature Extraction
   - TF-IDF
   - Word embeddings
   - Character n-grams

4. Model Training
   - Split data
   - Train model
   - Tune hyperparameters

5. Evaluation
   - Test accuracy
   - Check errors
   - Improve

6. Deployment
   - Save model
   - Create API
   - Monitor
""")

print("\n" + "=" * 70)
print("TEXT CLASSIFICATION SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. Text classification: Assign categories to text
2. TF-IDF + ML: Good baseline
3. CNN/LSTM: Better for complex patterns
4. BERT: State-of-the-art
5. Many applications: spam, intent, topic

Common Datasets:
- 20 Newsgroups
- IMDB Reviews
- SPAM dataset
- AG News
- SST-2

Next Steps:
- Try BERT for classification
- Build complete pipeline
- Deploy as API
""")
