"""
Text Preprocessing - NLP Basics
================================

This module covers:
- Text cleaning
- Tokenization
- Stopword removal
- Stemming and Lemmatization
- Regular expressions
"""

import numpy as np
import re

# Try to import NLTK
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not installed. Install with: pip install nltk")

# ============================================================================
# 1. TEXT CLEANING
# ============================================================================

print("=" * 70)
print("1. TEXT CLEANING")
print("=" * 70)

print("""
Text Cleaning:
- Convert to lowercase
- Remove special characters
- Remove numbers
- Remove extra whitespace
""")

sample_text = "Hello, World! This is a SAMPLE text with SOME 123 numbers and @special #characters!"

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

print(f"Original: {sample_text}")
print(f"Cleaned:  {clean_text(sample_text)}")

# More comprehensive cleaning
def clean_text_detailed(text):
    steps = []

    # Original
    steps.append(('Original', text))

    # Lowercase
    text = text.lower()
    steps.append(('Lowercase', text))

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    steps.append(('Remove URLs', text))

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    steps.append(('Remove HTML', text))

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    steps.append(('Remove special chars', text))

    # Remove extra whitespace
    text = ' '.join(text.split())
    steps.append(('Remove whitespace', text))

    return text

# Demo with HTML
html_text = "<p>This is a <b>sample</b> text with a <a href='http://example.com'>link</a></p>"
print(f"\nHTML Example:")
print(f"Original: {html_text}")
print(f"Cleaned:  {clean_text_detailed(html_text)}")

# ============================================================================
# 2. TOKENIZATION
# ============================================================================

print("\n" + "=" * 70)
print("2. TOKENIZATION")
print("=" * 70)

print("""
Tokenization: Split text into smaller units (tokens)
- Word tokenization: Split into words
- Sentence tokenization: Split into sentences
- Character tokenization: Split into characters
""")

text = "This is a sentence. This is another sentence! And one more?"

if NLTK_AVAILABLE:
    # Word tokenization
    words = word_tokenize(text)
    print(f"Word tokens: {words}")

    # Sentence tokenization
    sentences = sent_tokenize(text)
    print(f"Sentence tokens: {sentences}")

    # Download required data
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except:
        nltk.download('punkt_tab', quiet=True)
else:
    # Simple split-based tokenization
    words = text.split()
    print(f"Word tokens (simple): {words}")

    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"Sentence tokens (simple): {sentences}")

# ============================================================================
# 3. STOPWORD REMOVAL
# ============================================================================

print("\n" + "=" * 70)
print("3. STOPWORD REMOVAL")
print("=" * 70)

print("""
Stopwords: Common words that don't add meaning
- Examples: the, is, a, an, and, or, but

Why remove them:
- Reduce noise
- Speed up processing
- Focus on important words
""")

if NLTK_AVAILABLE:
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))

    text = "This is a sample text that demonstrates stopword removal"
    tokens = word_tokenize(text.lower())

    filtered = [word for word in tokens if word not in stop_words]

    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"After stopword removal: {filtered}")
    print(f"\nCommon stopwords: {list(stop_words)[:10]}")
else:
    # Simple stopword list
    stop_words = {'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    text = "This is a sample text that demonstrates stopword removal"
    words = text.lower().split()
    filtered = [w for w in words if w not in stop_words]

    print(f"Original: {text}")
    print(f"After stopword removal: {filtered}")

# ============================================================================
# 4. STEMMING
# ============================================================================

print("\n" + "=" * 70)
print("4. STEMMING")
print("=" * 70)

print("""
Stemming: Reduce words to root form
- Removes prefixes/suffixes
- Fast but imprecise
- Example: "running" → "run", "connected" → "connect"

Algorithms:
- Porter Stemmer (most common)
- Snowball Stemmer
- Lancaster Stemmer
""")

words = ["running", "runs", "runner", "ran", "connection", "connected", "connecting", "happiness", "happier", "happiest"]

if NLTK_AVAILABLE:
    try:
        nltk.data.find('corpora/wordnet')
    except:
        nltk.download('wordnet', quiet=True)

    stemmer = PorterStemmer()

    print("Porter Stemmer examples:")
    for word in words:
        stemmed = stemmer.stem(word)
        print(f"  {word:15} → {stemmed}")
else:
    print("NLTK not available for stemming demonstration")

# ============================================================================
# 5. LEMMATIZATION
# ============================================================================

print("\n" + "=" * 70)
print("5. LEMMATIZATION")
print("=" * 70)

print("""
Lemmatization: Reduce words to dictionary form
- Considers word context (POS tagging)
- More accurate than stemming
- Example: "running" → "run", "better" → "good"

Unlike stemming:
- Returns valid dictionary words
- Slower but more accurate
""")

words = ["running", "ran", "better", "good", "ate", "eating", "was", "were", "children", "child"]

if NLTK_AVAILABLE:
    lemmatizer = WordNetLemmatizer()

    print("WordNet Lemmatizer examples:")
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        print(f"  {word:15} → {lemma}")

    # Lemmatization with POS
    print("\nWith POS (verb):")
    for word in ["running", "ate", "gone"]:
        lemma = lemmatizer.lemmatize(word, pos='v')
        print(f"  {word:15} → {lemma} (verb)")
else:
    print("NLTK not available for lemmatization demonstration")

# ============================================================================
# 6. REGULAR EXPRESSIONS
# ============================================================================

print("\n" + "=" * 70)
print("6. REGULAR EXPRESSIONS")
print("=" * 70)

print("""
Regular Expressions: Pattern matching for text

Common patterns:
- \d: digit [0-9]
- \w: word character [a-zA-Z0-9_]
- \s: whitespace
- +: one or more
- *: zero or more
- ?: optional
- []: character class
""")

text = "Contact me at john@example.com or call 555-123-4567. My ID is #12345."

print(f"Original: {text}")
print(f"\nExtraction examples:")

# Extract email
emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
print(f"  Emails: {emails}")

# Extract phone
phones = re.findall(r'\d{3}-\d{3}-\d{4}', text)
print(f"  Phone numbers: {phones}")

# Extract hashtags
hashtags = re.findall(r'#\w+', text)
print(f"  Hashtags: {hashtags}")

# Extract numbers
numbers = re.findall(r'\d+', text)
print(f"  Numbers: {numbers}")

# ============================================================================
# 7. COMPLETE PREPROCESSING PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("7. COMPLETE PREPROCESSING PIPELINE")
print("=" * 70)

def preprocess_text(text, remove_stopwords=True, stem=False):
    """Complete text preprocessing pipeline"""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenize
    words = text.split()

    # Remove stopwords
    if remove_stopwords and NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        except:
            pass

    # Stemming
    if stem and NLTK_AVAILABLE:
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    return words

# Test the pipeline
sample = "The quick brown fox jumps over the lazy dog! Visit https://example.com for more info."

print(f"Original: {sample}")
print(f"\nAfter preprocessing:")
print(f"  Without stopwords: {preprocess_text(sample, remove_stopwords=True, stem=False)}")
print(f"  With stemming:    {preprocess_text(sample, remove_stopwords=True, stem=True)}")

# ============================================================================
# 8. HANDLING EMOJIS AND SPECIAL TEXT
# ============================================================================

print("\n" + "=" * 70)
print("8. HANDLING EMOJIS AND SPECIAL TEXT")
print("=" * 70)

text_with_emoji = "I love this! 😊 Great job! 🎉 #winning"

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

print(f"With emoji: {text_with_emoji}")
print(f"After removing emoji: {remove_emoji(text_with_emoji)}")

# ============================================================================
# 9. TEXT NORMALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("9. TEXT NORMALIZATION")
print("=" * 70)

print("""
Text Normalization: Standardize text variations

Common techniques:
- Contraction expansion: "don't" → "do not"
- Repeated characters: "loooove" → "loove"
- Slang: "gonna" → "going to"
- Abbreviations: "U.S.A." → "USA"
""")

# Handle contractions
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to"
}

text = "I don't know what's gonna happen. You're gonna love this!"

def expand_contractions(text):
    for contraction, expansion in contractions.items():
        text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
    return text

print(f"Original: {text}")
print(f"Expanded: {expand_contractions(text)}")

# Handle repeated characters
def remove_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

text_repeated = "This is soooooo coooooool!!!"
print(f"\nRepeated chars example:")
print(f"Original: {text_repeated}")
print(f"Fixed:    {remove_repeated_chars(text_repeated)}")

# ============================================================================
# 10. PRACTICAL EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("10. PRACTICAL EXAMPLE")
print("=" * 70)

# Sample reviews
reviews = [
    "This product is AMAZING!!! I love it so much 😊 #best",
    "Terrible quality. Don't buy this. Worst purchase ever 😡",
    "It's okay, not great not terrible. Average product.",
    "Just received my order. Works perfectly! Highly recommend.",
    "What a waste of money. Complete scam 😤 #fraud"
]

print("Preprocessing reviews:")
print("-" * 50)

for review in reviews:
    processed = preprocess_text(review)
    print(f"Original:  {review}")
    print(f"Processed: {processed}")
    print()

print("\n" + "=" * 70)
print("TEXT PREPROCESSING SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. Clean text: lowercase, remove special chars
2. Tokenize: split into words/sentences
3. Remove stopwords: filter common words
4. Stemming: fast but imprecise
5. Lemmatization: accurate dictionary form
6. Regex: powerful pattern matching

Preprocessing Pipeline:
1. Lowercase
2. Remove URLs, HTML
3. Remove special characters
4. Tokenize
5. Remove stopwords
6. Stem/Lemmatize
7. Join tokens

Next Steps:
- Learn text representation (BoW, TF-IDF)
- Apply to real NLP tasks
""")
