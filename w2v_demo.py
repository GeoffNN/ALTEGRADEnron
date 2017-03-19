# Adapted from Lab 5 : Word embeddings – unsupervised document classification
import os
import numpy as np
from time import time
import pickle as pkl
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from src.preprocess import body_dict_from_panda, get_email_ids_per_sender, get_conversation_ids, get_all_senders
from src.scoring import get_train_val, compute_prediction_mad
from src.postprocess import write_results_ranked
from src.tfidftools import get_tokens
from src.w2vtools import avg_word2vec, clean_string, compute_recommendations

warnings.filterwarnings('ignore')
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# % of most occuring words discarded as stop words
percentage_threshold = 0

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

# Compute stop words

stop_words = []
count_vec = CountVectorizer(tokenizer = get_tokens)
count_bow = count_vec.fit_transform(train_bodies.values())

voc_stats=np.sum(count_bow,0).T
features = count_vec.get_feature_names()

sorted_voc = [x for (y,x) in sorted(zip(voc_stats,features),reverse = True)]
stop_words = sorted_voc[:int(0.01*percentage_threshold*len(sorted_voc))]

# Compute average word2vec embeddings, load them if already computed
if not os.path.exists(path_to_data+'embeddings_stopwords_%s.p' % percentage_threshold):
    all_tokens = {}
    print("Tokenizing All text...")
    for i, key in enumerate(body_dict.keys()):
        all_tokens[key] = [token for token in get_tokens(clean_string(body_dict[key])) if token not in stop_words]
        if i % 1e3 == 0:
            print(i, 'emails processed')

    # create empty word vectors for the words in vocabulary
    # we set size=300 to match dim of GNews word vectors
    mcount = 5
    vectors = Word2Vec(size=3e2, min_count=mcount)
    # build vocabulary for 'vectors' from the list of lists of tokens with the build_vocab() method
    vectors.build_vocab(all_tokens.values())

    start = time()
    print("Loading Google News Word2Vec...")
    # we load only the Google word vectors corresponding to our vocabulary
    vectors.intersect_word2vec_format(path_to_data + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
    # normalize the vectors
    vectors.init_sims(replace=True)
    print("word vectors loaded and normalized. Took %s s" % (time() - start))

    print("Computing avg_word_embeddings...")
    start = time()
    all_embeddings = avg_word2vec(body_dict, vectors)
    print("Done. Took %s s" % (time() - start))

    with open(path_to_data+'embeddings_stopwords_%s.p' % percentage_threshold, 'wb') as f:
        pkl.dump(all_embeddings, f)

else:
    with open(path_to_data + 'embeddings_stopwords_%s.p' % percentage_threshold, 'rb') as f:
        all_embeddings = pkl.load(f)

# Use nearest centroid of average embeddings to compute recipient recommendations
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
                                              val_ids_per_sender, all_embeddings)

print("Done!")

print("Computing recommendations for train/test...")

n_recipients = 10

train_ids_per_sender = get_email_ids_per_sender(training)
conversation_ids = get_conversation_ids(train_ids_per_sender, training_info)
test_ids_per_sender = get_email_ids_per_sender(test)

recommendations_test = compute_recommendations(n_recipients, senders, training_info, conversation_ids, test_info,
                                               test_ids_per_sender, all_embeddings)
print("Done!")
print("Computing score on validation set...")
print("Score: {}".format(compute_prediction_mad(recommendations_val, val_info)))

print("Writing file...")
write_results_ranked(recommendations_val, path_to_results, "centroids_w2v_sw%s_validation.csv"%percentage_threshold)
write_results_ranked(recommendations_test, path_to_results, "centroids_w2v_sw%s_test.csv"%percentage_threshold)

with open(path_to_results + 'centroids_w2v_sw%s.p'%percentage_threshold, 'wb') as f:
    pkl.dump(recommendations_val, f)