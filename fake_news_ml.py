import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure NLTK data is downloaded
nltk.download("stopwords")
nltk.download("wordnet")

print("⏳ Loading data...")
# Load data (Ensure these paths match your folder structure)
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Labeling: 0 = Fake, 1 = Real
fake["label"] = 0
real["label"] = 1

# Combine and shuffle
df = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)
df["content"] = df["title"] + " " + df["text"]

# Cleaning Configuration
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Convert to string to avoid errors with NaNs
    text = str(text).lower()
    # Remove URLs and special characters
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    # Remove stopwords
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

print("🧹 Cleaning text data (this may take a moment)...")
df["content"] = df["content"].apply(clean_text)

# TF-IDF Vectorization
# UPGRADE: Added ngram_range=(1,2) to capture 2-word phrases (Bi-grams)
# UPGRADE: Added max_features=50000 to keep the model fast and prevent memory errors
print("🧠 Vectorizing with N-Grams (Unigrams + Bigrams)...")
X = df["content"]
y = df["label"]

vectorizer = TfidfVectorizer(max_df=0.7, ngram_range=(1, 2), max_features=50000)
X_tfidf = vectorizer.fit_transform(X)

# Train model
print("dt Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# Save model & vectorizer
print("💾 Saving model files...")
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Success! Model and vectorizer saved.")