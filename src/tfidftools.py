import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy.sparse import vstack


# TfIDF utilities


def get_tfidf_vector(mid, df, tfidf_model):
    body = list(df.loc[df['mid'] == int(mid), 'body'])[0]
    vector = tfidf_model.transform([body])
    return vector


def get_tfidf_vectors(mids, df_info, tfidf_model):
    """Return numpy array of the tfidf representations of mails in mids list"""
    return vstack([get_tfidf_vector(mid, df_info, tfidf_model) for mid in mids])


def get_tokens(body):
    """Tokenizes an email"""
    body = nltk.word_tokenize(body)
    body = [word.lower() for word in body]
    return body


def get_tfidf(token_dict, min_df=0.001, max_df=0.10):
    tfidf = TfidfVectorizer(tokenizer=get_tokens, min_df=min_df, max_df=max_df)
    # if dic is not modified, keys and values are in the same order
    keys = list(token_dict.keys())
    values = token_dict.values()
    tfs = tfidf.fit_transform(values)
    return tfidf, tfs, keys


def sparse_norm(f_mail):
    return np.sqrt(f_mail.dot(f_mail.T))

def get_count(token_dict, min_df=0.001, max_df=0.10):
    count = CountVectorizer(tokenizer=get_tokens, min_df=min_df, max_df=max_df)
    # if dic is not modified, keys and values are in the same order
    keys = list(token_dict.keys())
    values = token_dict.values()
    counts = count.fit_transform(values)
    return count, counts, keys