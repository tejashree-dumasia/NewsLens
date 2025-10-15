# File: app_dl_enhanced.py

import streamlit as st
import pickle
import re
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Configuration ---
st.set_page_config(
    page_title="News Classifier Pro",
    page_icon="📰",
    layout="wide" # Use the full page width
)

# --- Load Saved Model, Tokenizer, and Encoder ---
# Use st.cache_resource to load these only once
@st.cache_resource
def load_assets():
    try:
        model = load_model('news_classifier_model.h5') # Or 'best_model.h5'
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tokenizer, label_encoder
    except FileNotFoundError:
        st.error("Model assets not found. Please run the training script first.")
        st.stop()

model, tokenizer, label_encoder = load_assets()

# --- Constants ---
MAX_LEN = 100 # This MUST be the same as used in training
CONFIDENCE_THRESHOLD = 0.70 # Set a threshold for "confident" predictions

# --- Sidebar ---
st.sidebar.title("📰 News Classifier Pro")
st.sidebar.write("---")
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Deep Learning model (LSTM/CNN) to classify news articles "
    "into four categories: World, Sports, Business, or Sci/Tech."
)

st.sidebar.header("Model Details")
with st.sidebar.expander("Click to see details"):
    st.markdown("- **Model Architecture**: Bidirectional LSTM (example)")
    st.markdown("- **Vocabulary Size**: 10,000")
    st.markdown("- **Max Sequence Length**: 100")
    st.markdown(
        "Built with TensorFlow/Keras and deployed with Streamlit."
    )
st.sidebar.write("---")


# --- Main App Interface ---
st.title("Deep Learning News Classifier")
st.write("Enter a news headline or a short article below, or use one of the examples.")

# --- Example Buttons ---
st.subheader("Try an Example")
col1, col2, col3, col4 = st.columns(4)

# Use session state to store the text
if 'text' not in st.session_state:
    st.session_state.text = ""

if col1.button('Business 📈'):
    st.session_state.text = "The stock market saw a significant surge today as tech giants reported record profits."
if col2.button('Sci/Tech 🔬'):
    st.session_state.text = "Scientists have discovered a new exoplanet that could potentially support life, orbiting a nearby star."
if col3.button('Sports 🏅'):
    st.session_state.text = "In a stunning upset, the underdog team clinched the championship in the final seconds of the game."
if col4.button('World 🌎'):
    st.session_state.text = "International leaders gathered for a summit to discuss climate change and global economic policy."


# --- User Input ---
user_input = st.text_area(
    "Enter news text here:", 
    value=st.session_state.text, 
    height=150,
    key="user_input_area" # Add a key to help Streamlit
)

if st.button("Classify", type="primary"):
    if user_input:
        # Start a timer
        start_time = time.time()

        # 1. Preprocess and predict
        cleaned_input = re.sub('[^a-zA-Z]', ' ', user_input).lower()
        sequence = tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
        
        prediction_probs = model.predict(padded_sequence)[0]
        predicted_class_index = np.argmax(prediction_probs)
        predicted_confidence = prediction_probs[predicted_class_index]

        # Stop the timer
        end_time = time.time()
        processing_time = end_time - start_time

        # 2. Decode the prediction
        predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]

        # --- Display Results ---
        st.write("---")
        st.header("Results")
        
        # Display the main prediction
        st.subheader("Predicted Category:")
        st.success(f"**{predicted_class_name}** (Confidence: {predicted_confidence:.2%})")

        # Display a warning if confidence is low
        if predicted_confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"Low Confidence Warning: The model's confidence is below the {CONFIDENCE_THRESHOLD:.0%} threshold. The result may be less reliable.")

        # Display the confidence chart
        st.subheader("Confidence Scores:")
        prob_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability': prediction_probs
        })
        st.bar_chart(prob_df.set_index('Category'))
        
        # Display processing time
        st.info(f"Processing time: {processing_time:.4f} seconds.")

    else:
        st.warning("Please enter some text to classify.")