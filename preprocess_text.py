from nltk.tokenize import word_tokenize

def preprocess_text(text):
    stopwords = [
        "I",
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
