# File: app_dl.py

import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Saved Model, Tokenizer, and Encoder ---
try:
    model = load_model('news_classifier_model.h5')
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
except FileNotFoundError:
    st.error("Model, tokenizer, or encoder not found. Please run train_dl_model.py first.")
    st.stop()

# --- Constants ---
MAX_LEN = 100 # This MUST be the same as used in training

# --- Text Preprocessing Function ---
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    return " ".join(text.split())

# --- Streamlit App Interface ---
st.set_page_config(page_title="News Classifier (DL)", page_icon="📰")
st.title("📰 Deep Learning News Classifier")
st.write("Enter a news headline or a short description to classify it into one of four categories: World, Sports, Business, or Sci/Tech.")

# User input text area
user_input = st.text_area("Enter news text here:", height=150)

if st.button("Classify"):
    if user_input:
        # 1. Preprocess the input text
        cleaned_input = clean_text(user_input)

        # 2. Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        # 3. Predict the category
        prediction = model.predict(padded_sequence)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # 4. Decode the prediction to the original class name
        predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

        # 5. Display the result
        st.subheader("Prediction:")
        st.success(f"**The article is classified as: {predicted_class_name}**")
    else:
        st.warning("Please enter some text to classify.")