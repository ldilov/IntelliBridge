import nltk


def build_corpus():
    try:
        nltk.data.find("corpora/brown")
        nltk.data.find("corpora/webtext")
        nltk.data.find("corpora/gutenberg")
        nltk.data.find("corpora/reuters")
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("webtext", quiet=True)
        nltk.download("brown", quiet=True)
        nltk.download("gutenberg", quiet=True)
        nltk.download("reuters", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
    except Exception as ex:
        print(ex)

    from nltk.corpus import brown
    from nltk.corpus import gutenberg
    from nltk.corpus import reuters
    from nltk.corpus import webtext

    corpus_text = " ".join(
        [
            gutenberg.raw(fileid) for fileid in gutenberg.fileids()
        ] + [
            brown.raw(fileid) for fileid in brown.fileids()
        ] + [
            webtext.raw(fileid) for fileid in webtext.fileids()
        ] + [
            reuters.raw(fileid) for fileid in reuters.fileids()
        ]
    )

    return corpus_text
