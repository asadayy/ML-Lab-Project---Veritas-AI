import os
import re
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob

nltk.download("stopwords")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Model components
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec).max() * 100
    label = "REAL" if pred == 1 else "FAKE"
    
    sentiment = get_sentiment(text)
    
    return label, round(conf, 2), sentiment

# --- NEW: Explainability Function ---
def explain_prediction(text):
    """
    Identifies which words in the input contributed most to the decision.
    Returns: DataFrame of top contributors.
    """
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    # Get the index of words present in this specific text
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = vec.nonzero()[1]
    
    # Get coefficients for these specific words
    # Coef < 0 --> Fake Signal
    # Coef > 0 --> Real Signal
    word_impacts = []
    
    for idx in non_zero_indices:
        word = feature_names[idx]
        weight = model.coef_[0][idx]
        
        impact_type = "Fake Signal 🚨" if weight < 0 else "Real Signal ✅"
        word_impacts.append({
            "Word": word,
            "Impact Score": round(weight, 4),
            "Contribution": impact_type
        })
    
    # Create DataFrame and sort by absolute impact (ignoring zero)
    df = pd.DataFrame(word_impacts)
    if not df.empty:
        # Sort by magnitude (how strong the signal is)
        df['Abs_Impact'] = df['Impact Score'].abs()
        df = df.sort_values(by='Abs_Impact', ascending=False).drop(columns=['Abs_Impact'])
        
    return df.head(10) # Return top 10 most influential words