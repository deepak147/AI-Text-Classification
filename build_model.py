import pickle

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential


def build_model(
    X_train, X_val, y_train, y_val, model_w2v, tokenizer, label_encoder
):
    
    with open('max_len.txt', "rb") as file:
        max_length = int(file.read())
    # Build the deep learning model
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    for word, i in tokenizer.word_index.items():
        if word in model_w2v:
            embedding_matrix[i] = model_w2v[word]

    model = Sequential()
    model.add(
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=300,
            weights=[embedding_matrix],
            input_length=int(max_length),
            trainable=False,
        )
    )
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(4, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Define early stopping, hyperparameter
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Train the Model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
    )

    model_pkl_file = "chatgpt_model.pkl"
    with open(model_pkl_file, "wb") as file:
        pickle.dump(model, file)

    tokenizer_file = "tokenizer.pkl"
    with open(tokenizer_file, "wb") as file:
        pickle.dump(tokenizer, file)

    label_encoder_file = "label_encoder.pkl"
    with open(label_encoder_file, "wb") as file:
        pickle.dump(label_encoder, file)

    return model
