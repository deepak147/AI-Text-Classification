import pickle

import pandas as pd

from gensim.models import KeyedVectors

from preprocess_text import preprocess_text
from prepare_training_data import prepare_training_data
from build_model import build_model
from evaluate_model import evaluate_model
from predict import predict

df = pd.read_excel("ChatGPT Study Data.xlsx")
df["tokenized_text"] = df["text, cleaned"].apply(preprocess_text)

# Train Word2Vec model to generate word embeddings
word2vec_model_path = "GoogleNews-vectors-negative300.bin"
model_w2v = KeyedVectors.load_word2vec_format(
    word2vec_model_path, binary=True, limit=100000
)

# Prepare data for model training
X = df["tokenized_text"]
y = df["Author_type"]

# preprocess and prepare data for training
[
    X_train,
    X_val,
    y_train,
    y_val,
    tokenizer,
    max_length,
    label_encoder,
] = prepare_training_data(X, y, df)

max_len_file = "max_len.txt"
with open(max_len_file, "w") as file:
    file.write(str(max_length))

# build model using preprocessed data
model = build_model(
    X_train, X_val, y_train, y_val, model_w2v, tokenizer, label_encoder
)

# evaluate the model using testing results
evaluate_model(model, X_val, y_val, label_encoder)

# preidct results for new sentences
text = input("Input text to predict: ")
predict(text)
