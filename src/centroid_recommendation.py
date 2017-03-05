import pandas as pd
import numpy as np

from src.preprocess import get_email_ids_per_sender, get_all_senders, get_conversation_ids
from src.tfidftools import get_tfidf_vector, get_tfidf_vectors, sparse_norm

gamma = 1


def compute_recommendations(n_recipients, training, training_info, test, test_info, tfidf):
    train_ids_per_sender = get_email_ids_per_sender(training)
    test_ids_per_sender = get_email_ids_per_sender(test)
    conversation_ids = get_conversation_ids(train_ids_per_sender, training_info)
    recommendations = {}
    senders = get_all_senders(training)
    print("Computing for {} senders".format(len(senders)))
    k = 0
    for sender in senders:
        k += 1
        if k % 10 == 0:
            print("{} of {} senders computed".format(k, len(senders)))
        print("Computing neighbor centroids for {}".format(sender))
        centroids = get_recipient_centroids(sender, conversation_ids, training_info, tfidf)
        mids_to_process = test_ids_per_sender[sender]
        print("Processing {} emails for {}".format(len(mids_to_process), sender))

        j = 0
        for mid in mids_to_process:
            print(mid)
            recommendations[mid] = recommend_for_mid(n_recipients, mid, centroids, test_info, tfidf)
            if j % 5 == 0:
                print("{}/{} emails processed".format(j, len(mids_to_process)))
            j += 1

    return recommendations


def get_recipient_centroids(sender, conversation_ids, training_info, tfidf):
    """Return dict of centroids by recipient"""
    recipients = get_recipients(sender, conversation_ids)
    centroids = {}
    for recipient in recipients:
        centroids[recipient] = centroid(get_tfidf_vectors(conversation_ids[sender][recipient], training_info, tfidf))
    return centroids


def recommend_for_mid(n_recipients, mid, centroids, test_info, tfidf_model):
    feat = get_tfidf_vector(mid, test_info, tfidf_model)
    return get_closest(n_recipients, centroids, feat).index


def get_closest(n, centroids_dict, from_feat):
    dists = {recipient: similarity(from_feat, to_feat) for recipient, to_feat in centroids_dict.items()}
    return pd.Series(dists).sort_values()[:n]


def centroid(mail_feats_array):
    mean = np.mean(mail_feats_array, axis=0)
    return mean


def get_recipients(sender, conversation_ids):
    return conversation_ids[sender].keys()


# TODO: use time diff in similarity computation
def similarity(f_mail1, f_mail2):
    return np.dot(f_mail1, f_mail2.T) / (sparse_norm(f_mail1) * sparse_norm(f_mail2))


def decay(time_diff):
    return gamma ** time_diff
