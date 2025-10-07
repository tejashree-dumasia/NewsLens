# File: train_and_save_model.py

import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Text Preprocessing Function ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# --- 1. Load and Prepare Data ---
df = pd.read_csv('train.csv')
df['text'] = df['Title'] + " " + df['Description']
class_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
df['class_name'] = df['Class Index'].map(class_map)
df = df[['text', 'class_name']]

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# --- 2. Feature Extraction and Model Training ---
X = df['cleaned_text']
y = df['class_name']

# Initialize and fit the vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Initialize and train the model
model = MultinomialNB()
model.fit(X_tfidf, y)

print("Model and vectorizer training complete.")

# --- 3. Save the Model and Vectorizer ---
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer have been saved as 'model.pkl' and 'tfidf_vectorizer.pkl'")