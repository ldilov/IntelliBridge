import re


def normalize_corpus_text(corpus_text):
    corpus_text = corpus_text.lower()
    corpus_text = re.sub(r"[^a-zA-Z\s]+", "", corpus_text)

    return corpus_text


def tokenize_corpus_text(corpus_text):
    from nltk import word_tokenize

    corpus_tokens = word_tokenize(corpus_text)
    return corpus_tokens


def clean_corpus_tokens(corpus_tokens):
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    corpus_tokens = [token for token in corpus_tokens if token not in stop_words]
    return corpus_tokens
