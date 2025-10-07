# File: app.py

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Load Saved Model and Vectorizer ---
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please run train_and_save_model.py first.")
    st.stop()


# --- Text Preprocessing Function (must be identical to the one used in training) ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


# --- Streamlit App Interface ---

st.set_page_config(page_title="News Classifier", page_icon="📰")
st.title("📰 News Article Classifier")
st.write("Enter a news headline or a short description to classify it into one of four categories: World, Sports, Business, or Sci/Tech.")

# User input text area
user_input = st.text_area("Enter news text here:", height=150)

if st.button("Classify"):
    if user_input:
        # 1. Preprocess the input text
        cleaned_input = preprocess_text(user_input)

        # 2. Vectorize the cleaned text
        input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        # 3. Predict the category
        prediction = model.predict(input_tfidf)[0]

        # 4. Display the result
        st.subheader("Prediction:")
        st.success(f"**The article is classified as: {prediction}**")
    else:
        st.warning("Please enter some text to classify.")