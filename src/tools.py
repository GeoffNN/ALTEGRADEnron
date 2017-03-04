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
