import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Download NLTK data (only once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load model and encoder
@st.cache_resource
def load_models():
    pipeline = joblib.load('models/best_pipeline.pkl')
    le = joblib.load('models/label_encoder.pkl')
    return pipeline, le

pipeline, le = load_models()
# clean text 
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
extra_stopwords = {'film', 'movie', 'character', 'scene', 'time', 'story', 'one', 'even', 'also', 'get', 'make', 'see', 'way', 'well', 'first', 'two', 'still', 'may', 'us', 'take', 'much', 'though', 'something', 'someone', 'could', 'would', 'say', 'show', 'part', 'thing', 'actually', 'really', 'little'}
stop_words.update(extra_stopwords)
lemmatizer = WordNetLemmatizer() 
def clean_text(text):
  text = text.lower()
  # remove html tags
  text = re.sub(r'<.*?>','',text)
  # remove special characters
  text = re.sub(r'[^a-zA-Z0-9\s]','',text)

  # Tokenize
  words = nltk.word_tokenize(text)
  # remove stop words and lemmatize
  words =[lemmatizer.lemmatize(word) for word in words if word not in stop_words] # Apply lemmatization
  return ' '.join(words)

# Streamlit UI
st.title("🎬 IMDB Review Sentiment Classifier")
st.write("Enter a movie review and I'll tell you if it's positive or negative!")

user_input = st.text_area("Your review:", height=150)

if st.button("Analyze"):
    if user_input:
        cleaned = clean_text(user_input)
        pred_num = pipeline.predict([cleaned])[0]
        sentiment = le.inverse_transform([pred_num])[0]
        # Confidence (only if the classifier supports predict_proba)
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = proba[pred_num]
        st.write(f"### Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.write("Please enter a review.")