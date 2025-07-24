import joblib

# Load saved model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_news(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "REAL" if prediction == 1 else "FAKE"

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a news article to check if it's real or fake:\n")
    result = predict_news(user_input)
    print("\nPrediction:", result)
