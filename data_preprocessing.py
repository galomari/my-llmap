import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
import pickle
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_txt_data(txt_file_path, num_words=10000, max_length=100):
    # Read the file
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    
    # Fit the tokenizer and transform texts to sequences
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    # Just to visualize the tokenization result
    for i, (original, sequence) in enumerate(zip(corpus, padded_sequences)):
        print("Original: ", original)
        print("Tokenized: ", sequence)

    # Assuming binary classification for simplicity
    labels = np.random.randint(2, size=len(padded_sequences))  # Replace with actual labels if available

    return padded_sequences, labels, tokenizer

def save_tokenizer(tokenizer, tokenizer_save_path='tokenizer.pickle'):
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

def train_model(train_data, train_labels, num_words=10000, max_length=100, embedding_dim=16, model_save_path='my_model.h5'):
    model = Sequential([
        Embedding(num_words, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    try:
        history = model.fit(
            train_data, 
            train_labels, 
            epochs=10,
            validation_split=0.2,
            verbose=2
        )

        # Save the model
        model.save(os.path.join(r"C:\Users\hp\Downloads\models"))
        logger.info(f"Model trained successfully and saved to {model_save_path}")
    except Exception as e:
        logger.error(f"Failed to train the model. Error: {str(e)}")
        return None, None
    
    return model, history