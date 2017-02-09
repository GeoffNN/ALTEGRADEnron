import json
import nltk
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def timeit(f):
    """Decorator to time other function"""
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:{funcname} took: {time:.4f} sec'.format(
              funcname=f.__name__, time=te-ts))
        return result
    return timed


def dict_to_file(file_path):
    with open(file_path, 'w') as outfile:
        json.dump(token_dict, outfile)
    print('Done writing dict to file!')


def dict_from_file(file_path):
    with open(file_path) as infile:
        data = json.load(infile)
    return data


def body_dict_from_panda(dataframe):
    """Constructs dictionnary of bodies from dataframe with mid as key"""
    body_dict = {}
    nb_total = len(dataframe)
    print('Constructing dictionnary from dataframe...')
    for id, row in dataframe.iterrows():
        if(id % 10000 == 0):
            print('{id} / {nb_total}'.format(id=id, nb_total=nb_total))
        body_dict[row.mid] = row.body
    print('done !')
    return body_dict


def get_tokens(body):
    """Tokenizes an email"""
    body = nltk.word_tokenize(body)
    body = [word.lower() for word in body]
    return body


@timeit
def get_tfidf(token_dict, min_df=0.001, max_df=0.10):
    tfidf = TfidfVectorizer(tokenizer=get_tokens, min_df=min_df, max_df=max_df)
    tfs = tfidf.fit_transform(token_dict.values())
    return tfidf
