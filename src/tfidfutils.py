import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# TfIDF utilities


def get_tokens(body):
    """Tokenizes an email"""
    body = nltk.word_tokenize(body)
    body = [word.lower() for word in body]
    return body


def get_tfidf(token_dict, min_df=0.001, max_df=0.10):
    tfidf = TfidfVectorizer(tokenizer=get_tokens, min_df=min_df, max_df=max_df)
    tfs = tfidf.fit_transform(token_dict.values())
    return tfidf
