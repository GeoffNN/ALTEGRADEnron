import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
from src.tfidftools import get_tfidf_vector, get_tfidf_vectors, sparse_norm


def compute_recommendations(n_recipients, senders, training_info, conversation_ids, test_info, test_ids_per_sender,
                            tfidf):
    """Return recommendation dict : {mid: [rec1, rec2...] }"""
    recommendations = {}
    print("Computing for {} senders".format(len(senders)))
    pbar = tqdm(senders)
    for sender in pbar:
        centroids = get_recipient_centroids(sender, conversation_ids, training_info, tfidf)
        mids_to_process = test_ids_per_sender[sender]
        if len(mids_to_process) < 1:
            continue
        print("Processing {} emails for {}".format(len(mids_to_process), sender))
        for mid in mids_to_process:
            recommendations[mid] = recommend_for_mid(n_recipients, mid, centroids, test_info, tfidf)

    return recommendations


def get_recipient_centroids(sender, conversation_ids, training_info, tfidf):
    """Return dict of centroids by recipient"""
    recipients = get_recipients(sender, conversation_ids)
    centroids = {}
    for recipient in recipients:
        centroids[recipient] = centroid(get_tfidf_vectors(conversation_ids[sender][recipient], training_info, tfidf))
    return centroids


def get_recipients(sender, conversation_ids):
    return list(conversation_ids[sender].keys())


def recommend_for_mid(n_recipients, mid, centroids, test_info, tfidf_model):
    feat = get_tfidf_vector(mid, test_info, tfidf_model)
    # print(feat)
    return get_closest(n_recipients, centroids, feat).index


def get_closest(n, centroids_dict, from_feat):
    """n = int or 'max' - if 'max', return the whole ranking"""
    sims = {}
    for recipient, to_feat in centroids_dict.items():
        sims[recipient] = similarity(from_feat, to_feat)
    if n == 'max':
        return pd.Series(sims).sort_values(ascending=False)
    return pd.Series(sims).sort_values(ascending=False)[:n]


def centroid(mail_feats_array):
    mean = np.mean(mail_feats_array, axis=0)
    return mean


# TODO: use time diff in similarity computation
def similarity(f_mail1, f_mail2):
    if (sparse_norm(f_mail1) * sparse_norm(f_mail2)) == 0:
        return 0
    return float(f_mail1.dot(f_mail2.T) / (sparse_norm(f_mail1) * sparse_norm(f_mail2)))


def decay(time_diff, gamma=1):
    return gamma ** time_diff
