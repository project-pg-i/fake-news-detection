import streamlit as st
import joblib
import os
import re

# === Clean Function (must match train.py) ===
def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# === Load Model and Vectorizer ===
model_path = "models/logistic_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model or vectorizer file not found in 'models/' directory.")
    st.stop()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# === Streamlit UI ===
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️", layout="centered")
st.title("📰 Fake News Detection App")
st.subheader("Enter a news article below to check if it's **REAL** or **FAKE**:")

user_input = st.text_area("📝 Paste your news article here:", height=200)

# === Predict Button ===
if st.button("🔍 Check Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news article.")
    else:
        cleaned_text = clean(user_input)
        input_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        confidence = round(max(proba) * 100, 2)

        if prediction == 1:
            st.success(f"✅ This article appears to be **REAL** ({confidence}% confidence).")
        else:
            st.error(f"🚨 This article appears to be **FAKE** ({confidence}% confidence).")

# === Footer ===
st.markdown("---")
st.markdown("🔬 Built with Logistic Regression + TF-IDF | [GitHub Repo](https://github.com/project-pg-i/fake-news-detection)")
