import pandas as pd
import joblib
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Clean text function
def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load and prepare datasets
def load_datasets():
    print("📥 Loading datasets...")
    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")
    manual_df = pd.read_csv("data/manual_samples.csv")

    fake_df['label'] = 0  # Fake
    true_df['label'] = 1  # Real

    manual_df = manual_df[['title', 'text', 'label']]

    df = pd.concat([fake_df, true_df, manual_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"🗂️ Dataset shape: {df.shape}")
    print("📌 Label distribution:\n", df['label'].value_counts())
    print("🔍 Sample row:\n", df.iloc[0])

    return df

# Train model
def train_model(X_train, y_train):
    print("🎯 Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Main script
def main():
    df = load_datasets()

    # Combine and clean text
    print("🧹 Cleaning and combining text...")
    X = (df['title'] + " " + df['text']).apply(clean)
    y = df['label']

    # Train-test split
    print("🧪 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF vectorization
    print("🔠 Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("📊 TF-IDF shape (train):", X_train_tfidf.shape)
    print("📊 TF-IDF shape (test):", X_test_tfidf.shape)

    # Train and evaluate
    model = train_model(X_train_tfidf, y_train)

    print("📈 Evaluating model...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("\n📉 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("💾 Model and vectorizer saved to /models/")

if __name__ == "__main__":
    main()
