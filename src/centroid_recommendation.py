import pandas as pd
import numpy as np

from src.preprocess import body_dict_from_panda, get_email_ids_per_sender, get_all_senders, get_conversation_ids
from src.tfidftools import get_tfidf_vector, get_tfidf_vectors, sparse_norm

gamma = 1


def compute_recommendations(n_recipients, training, training_info, test, test_info, tfidf):
    train_ids_per_sender = get_email_ids_per_sender(training)
    test_ids_per_sender = get_email_ids_per_sender(test)
    conversation_ids = get_conversation_ids(train_ids_per_sender, training_info)
    recommendations = {}
    senders = get_all_senders(training)
    k = 0
    for sender in senders:
        k += 1
        if k % 100 == 0:
            print("{} of {} senders computed".format(k, len(senders)))
        recommendations[sender] = {}
        mids_to_process = test_ids_per_sender[sender]
        for mid in mids_to_process:
            recommendations[sender][mid] = recommend_for_sender(n_recipients, mid, sender, conversation_ids,
                                                                training_info, test_info, tfidf)
    return recommendations


def recommend_for_sender(n_recipients, mid, sender, conversation_ids, train_info, test_info, tfidf_model):
    centroids = get_recipient_centroids(sender, conversation_ids, train_info, tfidf_model)
    feat = get_tfidf_vector(mid, test_info, tfidf_model)
    return get_closest(n_recipients, centroids, feat).index


def get_recipient_centroids(sender, conversation_ids, training_info, tfidf):
    """Return dict of centroids by recipient"""
    recipients = get_recipients(sender, conversation_ids)
    centroids = {}
    for recipient in recipients:
        centroids[recipient] = centroid(get_tfidf_vectors(conversation_ids[sender][recipient], training_info, tfidf))
    return centroids


def get_recipients(sender, conversation_ids):
    return conversation_ids[sender].keys()


def centroid(mail_feats_array):
    mean = np.mean(mail_feats_array, axis=0)
    return mean


# TODO: use time diff in similarity computation
def similarity(f_mail1, f_mail2):
    return np.dot(f_mail1, f_mail2.T) / (sparse_norm(f_mail1) * sparse_norm(f_mail2))


def get_closest(n, centroids_dict, from_feat):
    dists = {recipient: similarity(from_feat, to_feat) for recipient, to_feat in centroids_dict.items()}
    return pd.Series(dists).sort_values()[:n]


def decay(time_diff):
    return gamma ** time_diff


