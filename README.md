# 📰 Fake News Detection

A machine learning project that classifies news articles as **REAL** or **FAKE** using TF-IDF vectorization and Logistic Regression. It includes a web interface built with **Streamlit** where users can test articles in real-time.

---

## 🚀 Demo

Try the app locally:
```bash
streamlit run app.py
```
📁 Project Structure

```
fake-news-detection/
├── data/                 # Contains Fake.csv, True.csv, and manual_samples.csv
├── models/               # Saved TF-IDF vectorizer and trained Logistic Regression model
├── src/                  # Source code
│   ├── train.py          # Model training script
│   └── predict.py        # CLI prediction script
├── app.py                # Streamlit web app
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

📦 Setup Instructions
1. Clone the repo
```bash
git clone https://github.com/project-pg-i/fake-news-detection.git
cd fake-news-detection
```
2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. pip install -r requirements.txt
```bash
python src/train.py
```
5. Launch Streamlit app
```bash
streamlit run app.py
```

📊 Model Details

- Vectorizer: TF-IDF (max_df=0.7, stop_words='english')
- Classifier: Logistic Regression (max_iter=1000)
- Dataset: Combined real/fake news from Kaggle + manually labeled samples
- Accuracy: ~98.8% on test data

🧪 Example Predictions

> ✅ REAL
> Title: WHO Approves New Malaria Vaccine
> Text: The World Health Organization has officially approved a new vaccine for malaria...
> Prediction: REAL (64.0% confidence)

> 🚨 FAKE
> Title: Aliens Ban Internet on Earth
> Text: The Galactic Federation has announced that it will disable Earth’s internet...
> Prediction: FAKE (83.3% confidence)

📚 Dataset Sources

- [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Manually curated samples (data/manual_samples.csv)

🛠 Built With

- Python 3
- Scikit-learn
- Pandas
- Streamlit
- Joblib

🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

🔗 Links

🔬 [Project Repo](https://github.com/project-pg-i/fake-news-detection)  
📘 [Streamlit Documentation](https://docs.streamlit.io/)
