import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only the NLTK resources needed for this app

nltk.download("stopwords")
nltk.download("wordnet")

# Load the trained model pipeline and label encoder only once
@st.cache_resource
def load_models():
    pipeline = joblib.load("models/best_pipeline.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return pipeline, label_encoder

# Store loaded model objects
pipeline, label_encoder = load_models()

# Load default English stopwords
stop_words = set(stopwords.words("english"))

# Add custom stopwords that may not contribute much to sentiment prediction
extra_stopwords = {
    "film", "movie", "character", "scene", "time", "story", "one", "even",
    "also", "get", "make", "see", "way", "well", "first", "two", "still",
    "may", "us", "take", "much", "though", "something", "someone", "could",
    "would", "say", "show", "part", "thing", "actually", "really", "little"
}
stop_words.update(extra_stopwords)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
 
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Split the text into individual words
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Join cleaned words back into a single string
    return " ".join(words)

# Set basic page configuration
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬")
st.title("🎬 IMDB Review Sentiment Classifier")
st.write("Enter a movie review and get its predicted sentiment.")
user_input = st.text_area("Your review:", height=150)
if st.button("Analyze"):
    
    # Check that the input is not empty or just spaces
    if user_input.strip():
        try:
            cleaned_input = clean_text(user_input)
            pred_num = pipeline.predict([cleaned_input])[0]

            # Convert numeric prediction back to original sentiment label
            sentiment = label_encoder.inverse_transform([pred_num])[0]

            st.success(f"Predicted Sentiment: {sentiment}")

            # Show confidence only if the selected model supports predict_proba()
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba([cleaned_input])[0]
                confidence = proba[pred_num]
                st.write(f"Confidence: {confidence:.2f}")

        # Show any unexpected runtime error in the app instead of crashing
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # If the input box is empty, show a warning message
    else:
        st.warning("Please enter a review before clicking Analyze.")