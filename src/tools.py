import json
import numpy
import pandas as pd
import numpy as np
import time


def timeit(f):
    """Decorator to time other function"""
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:{funcname} took: {time:.4f} sec'.format(
              funcname=f.__name__, time=te - ts))
        return result
    return timed


def dict_to_file(dictionnary, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(dictionnary, outfile)
    print('Done writing dict to file!')


def dict_from_file(file_path):
    with open(file_path) as infile:
        data = json.load(infile)
    return data


def apk(actual, predicted, k=10):
    """
        Forked from https://github.com/benhamner/Metrics/
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
        A list of elements that are to be predicted (order doesn't matter)
        predicted : list
        A list of predicted elements (order does matter)
        k : int, optional
        The maximum number of predicted elements
        Returns
        -------
        score : double
        The average precision at k over the input lists
        """
    if len(predicted)>k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
        Forked from https://github.com/benhamner/Metrics/
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        Parameters
        ----------
        actual : list
        A list of lists of elements that are to be predicted
        (order doesn't matter in the lists)
        predicted : list
        A list of lists of predicted elements
        (order matters in the lists)
        k : int, optional
        The maximum number of predicted elements
        Returns
        -------
        score : double
        The mean average precision at k over the input lists
        """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
