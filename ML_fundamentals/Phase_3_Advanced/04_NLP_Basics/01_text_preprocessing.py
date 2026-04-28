"""
=================================================================
01 - TEXT PREPROCESSING: Cleaning and Preparing Text Data
=================================================================
Topics:
  1. Basic text cleaning
  2. Tokenization
  3. Stopword removal
  4. Stemming
  5. Lemmatization
  6. Complete preprocessing pipeline
  7. Regular expressions for text
=================================================================
Prerequisites: pip install nltk
  Run once: python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
=================================================================
"""

import re
import string
import warnings
warnings.filterwarnings("ignore")

# Try importing NLTK
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, LancasterStemmer
    from nltk.stem import WordNetLemmatizer
    # Download required data (silent)
    for pkg in ['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng']:
        nltk.download(pkg, quiet=True)
    HAS_NLTK = True
    print("✅ NLTK imported")
except ImportError:
    HAS_NLTK = False
    print("❌ NLTK not installed. Run: pip install nltk")

# ── Section 1: Basic Text Cleaning ────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 1: Basic Text Cleaning")
print("=" * 65)

sample_text = "I can't BELIEVE it!!! This is the BEST movie I've ever seen 🎬🎬 #amazing @director123 https://example.com"

print(f"\n  Original: '{sample_text}'\n")

# Step 1: Lowercase
text_lower = sample_text.lower()
print(f"  1. Lowercase:      '{text_lower}'")

# Step 2: Remove URLs
text_no_url = re.sub(r'https?://\S+|www\.\S+', '', text_lower)
print(f"  2. Remove URLs:    '{text_no_url}'")

# Step 3: Remove @mentions and #hashtags
text_no_mentions = re.sub(r'[@#]\w+', '', text_no_url)
print(f"  3. Remove @/#:     '{text_no_mentions}'")

# Step 4: Remove emojis and special chars
text_clean = re.sub(r'[^\w\s]', '', text_no_mentions)
print(f"  4. Remove special: '{text_clean}'")

# Step 5: Remove extra whitespace
text_final = ' '.join(text_clean.split())
print(f"  5. Clean spaces:   '{text_final}'")

# ── Section 2: Tokenization ──────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Tokenization")
print("=" * 65)

print("""
  Tokenization = splitting text into individual tokens (words/sentences)
""")

text = "Natural language processing is amazing! It helps computers understand human language."

# Method 1: Simple split
tokens_split = text.split()
print(f"  str.split():    {tokens_split[:6]}...")

# Method 2: NLTK word tokenize
if HAS_NLTK:
    tokens_nltk = word_tokenize(text)
    print(f"  word_tokenize(): {tokens_nltk[:6]}...")
    
    # Sentence tokenization
    sentences = sent_tokenize(text)
    print(f"\n  Sentence tokenization:")
    for i, sent in enumerate(sentences):
        print(f"    {i+1}. '{sent}'")

# Method 3: Regex tokenizer
tokens_regex = re.findall(r'\b\w+\b', text.lower())
print(f"\n  Regex tokenize:  {tokens_regex[:6]}...")

# ── Section 3: Stopword Removal ───────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Stopword Removal")
print("=" * 65)

print("""
  Stopwords = common words that carry little meaning:
    "the", "is", "at", "which", "and", "a", "an", "in", etc.
""")

if HAS_NLTK:
    stop_words = set(stopwords.words('english'))
    print(f"  Total English stopwords: {len(stop_words)}")
    print(f"  Examples: {list(stop_words)[:15]}")
    
    # Remove stopwords
    text = "This is a very good movie and I think it is the best one"
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w not in stop_words and w.isalpha()]
    
    print(f"\n  Original tokens:  {tokens}")
    print(f"  After stopwords:  {filtered}")
    print(f"  Removed {len(tokens) - len(filtered)} stopwords")
else:
    # Manual stopword removal
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                  "at", "to", "for", "of", "and", "or", "it", "this", "that",
                  "i", "my", "me", "we", "you", "he", "she", "they", "very"}
    text = "this is a very good movie and i think it is the best one"
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    print(f"\n  Original: {tokens}")
    print(f"  Filtered: {filtered}")

# ── Section 4: Stemming ──────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Stemming")
print("=" * 65)

print("""
  Stemming = reducing words to their root/stem by chopping suffixes
  Fast but rough — may produce non-words!
""")

words = ["running", "runner", "ran", "runs", "easily", "fairly",
         "connection", "connected", "connecting", "flies", "flying"]

if HAS_NLTK:
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    
    print(f"\n  {'Word':<15s} {'Porter':>12s} {'Lancaster':>12s}")
    print("  " + "-" * 41)
    for word in words:
        print(f"  {word:<15s} {porter.stem(word):>12s} {lancaster.stem(word):>12s}")
    
    print("\n  💡 Porter is the most widely used stemmer")
    print("     Lancaster is more aggressive")

# ── Section 5: Lemmatization ──────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: Lemmatization")
print("=" * 65)

print("""
  Lemmatization = reducing words to dictionary form (lemma)
  Slower but accurate — always produces real words!
""")

if HAS_NLTK:
    lemmatizer = WordNetLemmatizer()
    
    test_words = [
        ("running", "v"),   # verb
        ("ran", "v"),
        ("better", "a"),    # adjective
        ("flies", "n"),     # noun
        ("flies", "v"),     # verb
        ("geese", "n"),
        ("studies", "v"),
        ("studying", "v"),
    ]
    
    print(f"\n  {'Word':<12s} {'POS':>5s} {'Stemmed':>12s} {'Lemmatized':>12s}")
    print("  " + "-" * 45)
    for word, pos in test_words:
        stemmed = porter.stem(word)
        lemmatized = lemmatizer.lemmatize(word, pos=pos)
        print(f"  {word:<12s} {pos:>5s} {stemmed:>12s} {lemmatized:>12s}")
    
    print("""
  ✦ Lemmatization gives "better" → "good" (stem gives "better")
  ✦ Lemmatization gives "flies" → "fly" (stem gives "fli")
  ✦ But lemmatization needs POS tag for best results!
    """)

# ── Section 6: Complete Preprocessing Pipeline ───────────────────
print("=" * 65)
print("SECTION 6: Complete Preprocessing Pipeline")
print("=" * 65)


def preprocess_text(text, use_lemma=True):
    """Complete text preprocessing pipeline."""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove @mentions and #hashtags
    text = re.sub(r'[@#]\w+', '', text)
    
    # 5. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 7. Tokenize
    if HAS_NLTK:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    
    # 8. Remove stopwords
    if HAS_NLTK:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = {"the", "a", "an", "is", "are", "was", "in", "on",
                      "at", "to", "for", "of", "and", "or", "it", "this", "that"}
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    
    # 9. Stem or Lemmatize
    if HAS_NLTK:
        if use_lemma:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(w) for w in tokens]
        else:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(w) for w in tokens]
    
    # 10. Join back
    return ' '.join(tokens)


# Test the pipeline
test_texts = [
    "I absolutely LOVED this movie!! 🎬 Best I've seen in years!!! #amazing",
    "The food was terrible... Worst restaurant ever :( @manager",
    "Check out this new product at https://example.com !!! It's AMAZING!!!",
    "<p>Running and flying are <b>exciting</b> activities!!</p>",
]

print("\n  Preprocessing results:")
for text in test_texts:
    clean = preprocess_text(text)
    print(f"\n  Raw:   '{text}'")
    print(f"  Clean: '{clean}'")

# ── Section 7: Regex Patterns for Text ────────────────────────────
print("\n" + "=" * 65)
print("SECTION 7: Useful Regex Patterns")
print("=" * 65)

patterns = {
    "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "URL": r'https?://\S+',
    "Phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "HTML Tags": r'<[^>]+>',
    "Hashtag": r'#\w+',
    "Mention": r'@\w+',
    "Number": r'\b\d+\b',
}

sample = "Contact john@example.com or call 555-123-4567. Visit https://site.com #hello @user <b>bold</b> 42"

print(f"\n  Sample: '{sample}'\n")
for name, pattern in patterns.items():
    matches = re.findall(pattern, sample)
    print(f"  {name:<12s}: {matches}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. Always clean text before analysis (lowercase, remove noise)
  2. Tokenization splits text into processable units
  3. Remove stopwords to focus on meaningful words
  4. Stemming is fast but crude; Lemmatization is accurate
  5. Build a reusable preprocessing pipeline
  6. Regex is powerful for pattern matching in text

📋 Standard Pipeline:
  lowercase → remove_noise → tokenize → stopwords → lemmatize

📚 Next: 02_text_representation.py (BoW, TF-IDF)
""")
