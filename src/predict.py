# src/predict.py

import joblib
import re
import string

def clean(text):
    # Match your train.py cleaning as closely as possible
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # another punctuation cleaner
    text = text.strip()
    return text

# Load model & vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Input
text = input("Enter a news article to check if it's real or fake:\n")

# Clean + vectorize
cleaned_text = clean(text)
vectorized = vectorizer.transform([cleaned_text])

# Predict
pred = model.predict(vectorized)[0]
label = "REAL" if pred == 1 else "FAKE"

print(f"\n🧠 Prediction: {label}")
