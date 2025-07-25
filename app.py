import streamlit as st
import joblib
import pickle
import os

# === Load Model and Vectorizer ===
model_path = "models/logistic_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model or vectorizer file not found in 'models/' directory.")
    st.stop()

with open(model_path, "rb") as f:
    model = joblib.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = joblib.load(f)

# === Streamlit App UI ===
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️", layout="centered")
st.title("📰 Fake News Detection App")
st.subheader("Enter a news article below to check if it's **REAL** or **FAKE**:")

# === Input Area ===
user_input = st.text_area("Paste your news article here:", height=200)

# === Predict Button ===
if st.button("🔍 Check Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news article.")
    else:
        # Vectorize the input and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 0:
            st.success("✅ This news article appears to be **REAL**.")
        else:
            st.error("🚨 This news article appears to be **FAKE**.")

# === Footer ===
st.markdown("---")
st.markdown("🔬 Built with Logistic Regression and TF-IDF | [GitHub Repo](https://github.com/project-pg-i/fake-news-detection)")
