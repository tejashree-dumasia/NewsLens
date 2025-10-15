# File: train_dl_model.py

import pandas as pd
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. Load and Prepare Data ---
df = pd.read_csv('train.csv')
df['text'] = df['Title'] + " " + df['Description']
class_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
df['class_name'] = df['Class Index'].map(class_map)
df = df[['text', 'class_name']]

# Simple text cleaning
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    return " ".join(text.split())

df['cleaned_text'] = df['text'].apply(clean_text)

# --- 2. Tokenization and Sequencing ---
# Hyperparameters
VOCAB_SIZE = 10000  # Number of words to keep in the vocabulary
MAX_LEN = 100       # Max length of sequences
EMBEDDING_DIM = 16  # Dimension of the word embeddings

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_text'])

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

# Pad sequences to ensure uniform length
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# --- 3. Encode Labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['class_name'])
y = to_categorical(y_encoded)

# --- 4. Build the Neural Network ---
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax') # Output layer: units = number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model ---
EPOCHS = 10
BATCH_SIZE = 32
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

print("\nModel training complete.")

# --- 6. Save the Model, Tokenizer, and Label Encoder ---
# Save the trained model
model.save('news_classifier_model.h5')

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("Model, tokenizer, and label encoder have been saved.")