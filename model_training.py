import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
import logging
import sys
import io
import contextlib

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

        # Capture the model summary
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            model.summary()
        model_details = stream.getvalue()

        logger.info("Model trained successfully.")
        return model, history, model_details

    except Exception as e:
        logger.error(f"Failed to train the model. Error: {str(e)}")
        return None, None, None

    return model, history, model_details