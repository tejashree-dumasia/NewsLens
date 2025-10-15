# File: app_batch_classifier.py

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
    layout="wide"
)

# --- Load Assets (cached) ---
@st.cache_resource
def load_assets():
    try:
        model = load_model('news_classifier_model.h5')
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
MAX_LEN = 100

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

# --- UI Tabs ---
tab1, tab2 = st.tabs(["Single Classifier", "Batch Classifier"])

# --- TAB 1: Single Headline Classifier ---
with tab1:
    st.header("Classify a Single News Headline")
    user_input = st.text_area("Enter news text here:", height=150, key="single_input")

    if st.button("Classify Single", type="primary"):
        if user_input:
            # ... (Logic from the previous single classifier app) ...
            cleaned_input = re.sub('[^a-zA-Z]', ' ', user_input).lower()
            sequence = tokenizer.texts_to_sequences([cleaned_input])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
            
            prediction_probs = model.predict(padded_sequence)[0]
            predicted_class_index = np.argmax(prediction_probs)
            
            predicted_class_name = label_encoder.inverse_transform([predicted_class_index])[0]
            st.success(f"**Predicted Category: {predicted_class_name}**")
            
            prob_df = pd.DataFrame({
                'Category': label_encoder.classes_,
                'Probability': prediction_probs
            })
            st.bar_chart(prob_df.set_index('Category'))
        else:
            st.warning("Please enter some text.")

# --- TAB 2: Batch Classifier ---
with tab2:
    st.header("Classify a Batch of News Headlines")
    st.write("Enter one headline per line below.")

    batch_input = st.text_area("Paste headlines here:", height=250, key="batch_input")

    if st.button("Classify Batch", type="primary"):
        if batch_input:
            start_time = time.time()

            # 1. Split input text into a list of headlines
            headlines = [line for line in batch_input.strip().split('\n') if line.strip()]
            
            if not headlines:
                st.warning("Please enter at least one headline.")
            else:
                st.info(f"Found {len(headlines)} headlines to classify.")
                
                # 2. Preprocess and predict in a batch
                cleaned_headlines = [re.sub('[^a-zA-Z]', ' ', h).lower() for h in headlines]
                sequences = tokenizer.texts_to_sequences(cleaned_headlines)
                padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

                predictions = model.predict(padded_sequences)

                # 3. Process results
                results = []
                for i, headline in enumerate(headlines):
                    pred_index = np.argmax(predictions[i])
                    confidence = predictions[i][pred_index]
                    category = label_encoder.inverse_transform([pred_index])[0]
                    results.append({
                        "Headline": headline,
                        "Predicted Category": category,
                        "Confidence": f"{confidence:.2%}"
                    })

                end_time = time.time()
                st.success(f"Batch classification completed in {end_time - start_time:.2f} seconds.")

                # 4. Display results in a table
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                # 5. Add a download button for the results
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_data = convert_df_to_csv(results_df)
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv_data,
                    file_name='classification_results.csv',
                    mime='text/csv',
                )
        else:
            st.warning("The text area is empty. Please paste some headlines.")