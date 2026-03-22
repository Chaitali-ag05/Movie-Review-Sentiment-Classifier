# 🎬 IMDB Review Sentiment Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-review-sentiment-classifier.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning web application that predicts whether a movie review is **positive** or **negative** using Natural Language Processing (NLP). The app is built with Python, scikit‑learn, and Streamlit, and achieves **~88% accuracy** on the IMDB dataset.

👉 **[Live Demo](https://movie-review-sentiment-classifier-njwkfmi6buvpkn6yup7ydg.streamlit.app/)**  
👉 **[GitHub Repository](https://github.com/Chaitali-ag05/Movie-Review-Sentiment-Classifier)**

---

## ✨ Features

- **Real‑time sentiment analysis** – Paste any movie review and get instant feedback.
- **Text preprocessing pipeline** – HTML removal, tokenization, stopword removal, lemmatization.
- **Multiple models compared** – Logistic Regression, Naive Bayes, SVM (best: Logistic Regression with 88% accuracy).
- **Confidence scores** – See how confident the model is.
- **Interactive UI** – Built with Streamlit for a smooth user experience.

---

## 🧠 How It Works

1. **Text preprocessing** – The review is cleaned and lemmatized using NLTK.
2. **Feature extraction** – TF‑IDF vectorization converts text to numerical features.
3. **Prediction** – A trained Logistic Regression model classifies the sentiment.
4. **Result** – The app displays the sentiment (positive/negative) and confidence score.

---

## 📊 Model Performance

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | **88.2%** |
| SVM                 | 87.9%    |
| Naive Bayes         | 85.4%    |

The best model (Logistic Regression) was saved using `joblib` and is deployed in the app.

---

## 🛠️ Tech Stack

- **Python** 3.8+
- **scikit‑learn** – TF‑IDF vectorization, model training
- **NLTK** – Text preprocessing (stopwords, lemmatization, tokenization)
- **Streamlit** – Web app framework
- **joblib** – Model serialization
- **pandas** – Data handling
- **Git & GitHub** – Version control

## 📂 Dataset

This project uses the **IMDB Movie Reviews Dataset**, which contains **50,000 labeled movie reviews** for sentiment analysis.

- 25,000 positive  reviews 
- 25,000 negative reviews
- Balanced dataset with **positive** and **negative** sentiments

The dataset is widely used as a benchmark for **NLP sentiment classification tasks**.

---
## 📁 Project Structure

```text
Movie-Review-Sentiment-Classifier/
│
├── app.py                     # Streamlit application
├── models/
│   ├── best_pipeline.pkl      # Trained model + vectorizer
│   └── label_encoder.pkl      # Label encoder for sentiments
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files to ignore in Git
```
## 🚀 Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chaitali-ag05/Movie-Review-Sentiment-Classifier.git
   cd Movie-Review-Sentiment-Classifier

2. **Create a virtual environment (optional but recommended)**
      ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows: venv\Scripts\activate
3. ** Install dependencies **
   ```bash
    pip install -r requirements.txt
4.  **Run the Streamlit application**
    ```bash
    streamlit run app.py
5. **Open your browser to http://localhost:8501 **
---
## 🔮 Future Improvements

- Add deep learning models (LSTM / BERT)
- Improve UI and add review history
- Deploy using Docker or cloud platforms

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
---

## 👩‍💻 Author

**Chaitali Agrawal**  
B.Tech Student – Data Science & Computer Science Engineering
