# Adapted from Lab 5 : Word embeddings – unsupervised document classification
import re
import string
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity as cosine
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

print(sys.version)


# remove dashes and apostrophes from punctuation marks
punct = string.punctuation.replace('-', '').replace("'",'')
# regex to match intra-word dashes and intra-word apostrophes
my_regex = re.compile(r"(\b[-']\b)|[\W_]")


# performs basic pre-processing
# note: we do not lowercase for consistency with Google News embeddings
def clean_string(string, punct=punct, my_regex=my_regex):
    # remove formatting
    str = re.sub('\s+', ' ', string)
    # remove punctuation (preserving dashes)
    str = ''.join(l for l in str if l not in punct)
    # remove dashes that are not intra-word
    str = my_regex.sub(lambda x: (x.group(1) if x.group(1) else ' '), str)
    # strip extra white space
    str = re.sub(' +',' ',str)
    # strip leading and trailing white space
    str = str.strip()
    return str

def avg_word2vec(body_dict, model):
    """ Returns average word2vec representations dict : {mid : avg_w2v_representation} """
    embeddings = {}
    for key in body_dict.keys():
        avgword2vec = None
        for word in body_dict[key]:
            # get embedding (if it exists) of each word in the sentence
            if word in model.vocab:
                if avgword2vec is None:
                    avgword2vec = model[word]
                else:
                    avgword2vec = avgword2vec + model[word]

        # if at least one word in the sentence has a word embeddings :
        if avgword2vec is not None:
            avgword2vec = avgword2vec / len(avgword2vec)  # normalize sum
            embeddings[key] = avgword2vec
        else:
            embeddings[key] = np.zeros((300,))
    print('Generated embeddings for {0} sentences.'.format(len(embeddings)))
    return embeddings

def compute_recommendations(n_recipients, senders, training_info, conversation_ids, test_info, test_ids_per_sender,
                            feat_dict):
    """Returns recommendation dict : {mid: [rec1, rec2...] }"""
    recommendations = {}
    print("Computing for {} senders".format(len(senders)))
    pbar = tqdm(senders)
    for sender in pbar:
        centroids = get_recipient_centroids(sender, conversation_ids, feat_dict)
        mids_to_process = test_ids_per_sender[sender]
        if len(mids_to_process) < 1:
            continue
        print("Processing {} emails for {}".format(len(mids_to_process), sender))
        for mid in mids_to_process:
            recommendations[mid] = get_closest(n_recipients, centroids, feat_dict[mid]).index

    return recommendations


def get_recipient_centroids(sender, conversation_ids, feat_dict):
    """Return dict of centroids by recipient"""
    recipients = get_recipients(sender, conversation_ids)
    centroids = {}
    for recipient in recipients:
        centroids[recipient] = centroid(get_features(feat_dict, conversation_ids[sender][recipient]))
    return centroids

def get_features(feat_dict, keys):
    list_features = []
    for k in keys:
        list_features.append(feat_dict[k])
    return list_features

def get_recipients(sender, conversation_ids):
    return list(conversation_ids[sender].keys())


def get_closest(n, centroids_dict, from_feat):
    """n = int or 'max' - if 'max', return the whole ranking"""
    sims = {}
    for recipient, to_feat in centroids_dict.items():
        sims[recipient] = cosine(from_feat, to_feat)
    if n == 'max':
        return pd.Series(sims).sort_values(ascending=False)
    return pd.Series(sims).sort_values(ascending=False)[:n]


def centroid(mail_feats_array):
    mean = np.mean(mail_feats_array, axis=0)
    return mean