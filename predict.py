import pickle

import numpy as np

from preprocess_text import preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict(text):
    
    #encode the input according to the model requirements
    with open('max_len.txt', "r") as file:
        max_length = int(file.read())
    with open("chatgpt_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    with open("label_encoder.pkl", "rb") as file: 
        label_encoder = pickle.load(file)
    text = preprocess_text(text)
    new_sequences = tokenizer.texts_to_sequences([text])
    new_padded = pad_sequences(new_sequences, maxlen=int(max_length), padding="post")
    predictions = model.predict(new_padded)
    return predictions
