import pandas as pd

from src.centroid_recommendation import compute_recommendations
from src.preprocess import body_dict_from_panda
from src.tfidftools import get_tfidf

path_to_data = 'data/'

n_recipients = 10

##########################
# load some of the files #
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(
    path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(
    path_to_data + 'test_info.csv', sep=',', header=0)

# Compute useful structures

train_bodies = body_dict_from_panda(training_info)
test_bodies = body_dict_from_panda(test_info)

body_dict = {**train_bodies, **test_bodies}

print('Fitting tfidf, this will take some time...')
tfidf, tfs, keys = get_tfidf(body_dict, 0.001, 0.10)
print("Fitted.")

print("Computing recommendations...")
recommendations = compute_recommendations(n_recipients, training, training_info, test, test_info, tfidf)
print("Recommendations computed.")
print("Saving to file...")
print("All set!")
