from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


def prepare_training_data(X, y, df):
    # Encode the author types to numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Convert text data into numerical sequences using Word2Vec embeddings
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)

    # Pad sequences to have the same length
    max_length = max(len(seq) for seq in X_sequences)
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding="post")

    # Convert numerical labels to categorical
    y_categorical = to_categorical(y_encoded)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y_categorical, test_size=0.2, random_state=42
    )

    return [X_train, X_val, y_train, y_val, tokenizer, max_length, label_encoder]
