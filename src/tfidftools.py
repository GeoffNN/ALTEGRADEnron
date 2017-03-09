from collections import defaultdict
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np
from tqdm import tqdm_notebook


import src.textembeddingtools as texttools

# TfIDF utilities


def get_tfidf_vector(mid, df, tfidf_model):
    body = list(df.loc[df['mid'] == int(mid), 'body'])[0]
    vector = tfidf_model.transform([body])
    return vector


def get_tfidf_vectors(mids, df_info, tfidf_model):
    """Return numpy array of the tfidf representations of mails in mids list"""
    return vstack([get_tfidf_vector(mid, df_info, tfidf_model) for mid in mids])


def get_tfidf(token_dict, min_df=0.001, max_df=0.10):
    tfidf = TfidfVectorizer(tokenizer=texttools.tokenize_body,
                            min_df=min_df, max_df=max_df)
    # if dic is not modified, keys and values are in the same order
    keys = list(token_dict.keys())
    values = token_dict.values()
    tfs = tfidf.fit_transform(values)
    return tfidf, tfs, keys


def sparse_norm(f_mail):
    return np.sqrt(f_mail.dot(f_mail.T))


def get_tokens(body):
    return texttools.tokenize_body(body)


def get_idf_dic(token_dict, min_count=40, max_count=400):
    """
    removes words that appear less then @min_count times
    or more than @max_count
    """
    doc_count = len(token_dict)
    idf_dic = defaultdict(int)
    pbar_tokens = tqdm_notebook(token_dict.values())
    for tokens in pbar_tokens:
        unique_tokens = list(set(tokens))
        for token in unique_tokens:
            idf_dic[token] += 1

    idf_dic_log_without_rare = {word: math.log10(doc_count / count)
                                for word, count in idf_dic.items()
                                if count >= min_count and
                                count <= max_count}
    idf_words = list(idf_dic_log_without_rare.keys())
    return idf_dic_log_without_rare, idf_words
