import pandas as pd

from src.centroid_recommendation import compute_recommendations
from src.postprocess import write_results_ranked
from src.preprocess import body_dict_from_panda, get_email_ids_per_sender, get_conversation_ids, get_all_senders
from src.scoring import get_train_val, compute_prediction_mad
from src.tfidftools import get_tfidf, get_count
import pickle as pkl

path_to_data = 'data/'
path_to_results = 'results/'

n_recipients = 'max'

##########################
# load some of the files #
##########################
features = 'counts'
features = 'tfidf'

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(
    path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

test_info = pd.read_csv(
    path_to_data + 'test_info.csv', sep=',', header=0)

# Compute useful structures

train_info, train_ids_per_sender_val, val_info, val_ids_per_sender = get_train_val(training, training_info,
                                                                                   0.95, disp=True)

train_bodies = body_dict_from_panda(train_info)
val_bodies = body_dict_from_panda(val_info)

test_bodies = body_dict_from_panda(test_info)

body_dict = {**train_bodies, **val_bodies, **test_bodies}


if features == 'tfidf':
    print('Fitting tfidf, this will take some time...')
    vectorizer, feats, keys = get_tfidf(body_dict, 0.001, 0.10)
    print("Fitted.")

elif features == 'counts':
    print('Computing word counts, this will take some time...')
    vectorizer, feats, keys = get_count(body_dict, 0.001, 0.10)
    print("Fitted.")

print("Computing recommendations for train/val...")

conversation_ids_val = get_conversation_ids(train_ids_per_sender_val, training_info)
senders = get_all_senders(training)
recommendations_val = compute_recommendations(n_recipients, senders, train_info, conversation_ids_val, val_info,
                                              val_ids_per_sender, vectorizer)
print("Done!")

print("Computing recommendations for train/test...")

n_recipients = 10

train_ids_per_sender = get_email_ids_per_sender(training)
conversation_ids = get_conversation_ids(train_ids_per_sender, training_info)
test_ids_per_sender = get_email_ids_per_sender(test)

recommendations_test = compute_recommendations(n_recipients, senders, training_info, conversation_ids, test_info,
                                               test_ids_per_sender, vectorizer)
print("Done!")
print("Computing score on validation set...")
print("Score: {}".format(compute_prediction_mad(recommendations_val, val_info)))

print("Writing file...")
write_results_ranked(recommendations_val, path_to_results, "centroids_%s_validation.csv"%features)
write_results_ranked(recommendations_test, path_to_results, "centroids_%s_test.csv"%features)

with open(path_to_results + 'centroids_%s_dict.p'%features, 'wb') as f:
    pkl.dump(recommendations_val, f)