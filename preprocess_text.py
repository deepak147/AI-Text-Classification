from google.colab import drive
import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import shap

df = pd.read_ecxel('Two classes - ChatGPT Study Data.xlsx', engine = 'openpyxl')

def preprocess_text(text):
    stopwords = [
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "com",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "who",
        "will",
        "with",
        "the",
        "www",
    ]

    words = word_tokenize(text.lower())
    text = ' '.join([word for word in words if word.isalpha() and word not in stopwords])
    return text

df['text'] = df['text, cleaned'].apply(preprocess_text)


# Split data into labels and text
labels = df['author_type']
texts = df['text']
