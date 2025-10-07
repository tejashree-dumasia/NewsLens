import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load and Prepare Data ---

# Load the dataset
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please download it from the Kaggle link provided.")
    exit()


# Combine Title and Description into a single text column
df['text'] = df['Title'] + " " + df['Description']

# Map class index to human-readable names
class_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
df['class_name'] = df['Class Index'].map(class_map)

# Select relevant columns
df = df[['text', 'class_name']]

print("Dataset Head:")
print(df.head())
print("\nClass Distribution:")
print(df['class_name'].value_counts())


# --- 2. Text Preprocessing ---

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and preprocesses a single text entry."""
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    
    # Tokenize the text
    words = text.split()
    
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply preprocessing to the 'text' column
print("\nPreprocessing text data...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("Preprocessing complete.")


# --- 3. Feature Extraction (TF-IDF) ---

# Initialize the TF-IDF Vectorizer
# TF-IDF stands for Term Frequency-Inverse Document Frequency.
# It converts text into a matrix of numbers, giving more weight to words that are
# important to a document but not common across all documents.
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Use top 5000 features

# Define features (X) and target (y)
X = df['cleaned_text']
y = df['class_name']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit the vectorizer on the training data and transform both train and test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# --- 4. Model Training ---

# Initialize and train the Multinomial Naive Bayes model
print("\nTraining the Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("Model training complete.")


# --- 5. Model Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Display classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_map.values()))

# Display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_map.values(), yticklabels=class_map.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# --- 6. Function to Predict New Headlines ---

def classify_news(headline):
    """Takes a new headline string and predicts its category."""
    # Preprocess the headline
    cleaned_headline = preprocess_text(headline)
    
    # Vectorize the headline using the already-fitted vectorizer
    headline_tfidf = tfidf_vectorizer.transform([cleaned_headline])
    
    # Predict the category
    prediction = model.predict(headline_tfidf)
    
    return prediction[0]

# --- Example Usage ---
print("\n--- Testing with new headlines ---")
news1 = "NASA launches new mission to explore distant galaxies and stars"
print(f"Headline: '{news1}' \nPredicted Category: {classify_news(news1)}\n")

news2 = "Stock market surges as tech companies report record profits"
print(f"Headline: '{news2}' \nPredicted Category: {classify_news(news2)}\n")

news3 = "Team wins championship in a thrilling last-minute goal"
print(f"Headline: '{news3}' \nPredicted Category: {classify_news(news3)}\n")