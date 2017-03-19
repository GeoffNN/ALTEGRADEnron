import pandas as pd
import time

import src.tools as tools

# Load data

path_to_data = 'data/'

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(
    path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)


# Fit tfidf

body_dict = tools.body_dict_from_panda(training_info)

print('Fitting tfidf, this will take some time...')

tfidf = tools.get_tfidf(body_dict, 0.001, 0.10)


# Test on some random sentence

some_body = 'This a great day for us to do some tfidf, don\'t you think so ?'
res_tfidf = tfidf.transform([some_body])

print(some_body)
print('has the following tfidf representation :')
print(res_tfidf)
