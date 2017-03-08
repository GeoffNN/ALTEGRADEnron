import pandas as pd

from time import time

from src.centroid_recommendation import compute_recommendations
from src.postprocess import write_results_ranked
from src.preprocess import body_dict_from_panda, get_email_ids_per_sender, get_conversation_ids, get_all_senders
from src.scoring import get_train_val, compute_prediction_mad
import pickle as pkl

from src.tfidftools import get_tokens
from gensim.models.word2vec import Word2Vec


path_to_data = 'data/'
path_to_results = 'results/'

n_recipients = 'max'

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

train_info, train_ids_per_sender_val, val_info, val_ids_per_sender = get_train_val(training, training_info,
                                                                                   0.95, disp=True)

train_bodies = body_dict_from_panda(train_info)
val_bodies = body_dict_from_panda(val_info)

test_bodies = body_dict_from_panda(test_info)

body_dict = {**train_bodies, **val_bodies, **test_bodies}

##########################
# Compute features #
##########################
print("Tokenizing Train...")
for i, key in enumerate(train_bodies.keys()):
    train_bodies[key] = get_tokens(clean_string(train_bodies[key]))
    if i % 1e3 == 0:
        print(i, 'emails processed')

print("Tokenizing Validation...")
for i, key in enumerate(val_bodies.keys()):
    val_bodies[key] = get_tokens(clean_string(val_bodies[key]))
    if i % 1e3 == 0:
        print(i, 'emails processed')

print("Tokenizing Test...")
for i, key in enumerate(test_bodies.keys()):
    test_bodies[key] = get_tokens(clean_string(test_bodies[key]))
    if i % 1e3 == 0:
        print(i, 'emails processed')

# create empty word vectors for the words in vocabulary
# we set size=300 to match dim of GNews word vectors
mcount = 5
vectors = Word2Vec(size=3e2, min_count=mcount)
# build vocabulary for 'vectors' from the list of lists of tokens with the build_vocab() method
vectors.build_vocab(body_dict.values())

start = time()
print("Loading 20 News Group...")
# we load only the Google word vectors corresponding to our vocabulary
vectors.intersect_word2vec_format(path_to_data + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
# normalize the vectors
vectors.init_sims(replace=True)
print("word vectors loaded and normalized. Took %s s"%(time()-start))







##########################
# Proceed to knn on centroids #
##########################

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