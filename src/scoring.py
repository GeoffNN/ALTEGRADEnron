import numpy as np


import src.tools as tools
import src.preprocess as preprocess
import src.recencytools as recency


def compute_prediction_mad(prediction_dic, val_info):
    """
    @prediction_dic {mid:[recipient1, recipient2, ...], ...}
    @val_info dataframe with ground truth senders
    returns mad score
    """
    actual = []
    pred = []
    for mid, recipients in prediction_dic.items():
        real_recipients = preprocess.get_recipients(val_info, mid)
        actual.append(real_recipients)
        pred.append(recipients)
    score = mapk(actual, pred)
    return score


def apk(actual, predicted, k=10):
    """
        Forked from https://github.com/benhamner/Metrics/
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
        A list of elements that are to be predicted (order doesn't matter)
        predicted : list
        A list of predicted elements (order does matter)
        k : int, optional
        The maximum number of predicted elements
        Returns
        -------
        score : double
        The average precision at k over the input lists
        """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
        Forked from https://github.com/benhamner/Metrics/
        Computes the mean average precision at k.
        This function computes the mean average precision
        at k between two lists of lists of items.
        Parameters
        ----------
        actual : list
        A list of lists of elements that are to be predicted
        (order doesn't matter in the lists)
        predicted : list
        A list of lists of predicted elements
        (order matters in the lists)
        k : int, optional
        The maximum number of predicted elements
        Returns
        -------
        score : double
        The mean average precision at k over the input lists
        """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def compute_recency_prediction_mad(recency_predictions, val_info):
    """
    ** Prefer using recency.recency_predictions_to_standard
    and scoring.compute_prediction_mad **

    @recency_predictions dict {sender:[[mids,[senders, senders, ...]]}
    @val_info dataframe with ground truth senders
    returns mad score
    """
    actual = []
    pred = []
    for sender in recency_predictions:
        sender_mids = recency_predictions[sender][0]
        for mid in sender_mids:
            real_recipients = preprocess.get_recipients(val_info, mid)
            actual.append(real_recipients)
            pred.append(recency_predictions[sender][1][0])
    score = mapk(actual, pred)
    return score


def get_train_val(training, training_info, train_frac=.5, disp=True):
    """
    Creates cross-validation structures
    The train-val sets are split chronologically
    to respect the train-test setting
    @return train_info : train_info data frame (same format as training_info)
    @return val_info
    @return train_email_ids_per_sender {sender: [mid1, mid2, ...]}
    @return val_email_ids_per_sender
    """
    # Sort the training mails chronologically
    training_info = recency.add_time_rank_to_dataframe(training_info)

    # Compute training and validation email numbers
    n_samples = len(training_info)
    n_train = int(train_frac * n_samples)
    n_test = n_samples - n_train

    # Training info has to be chronologically ordered before this step!
    train_info = training_info.head(n_train)
    val_info = training_info.tail(n_test)

    # Construct list of train and val mids
    train_mids = np.array(train_info.mid)
    val_mids = np.array(val_info.mid)

    # Extract {sender : mids} dicts
    if(disp):
        print('Processing training !')

    val_email_ids_per_sender = preprocess.get_restricted_email_ids_per_sender(
        training, val_mids)

    if(disp):
        print('Processing val !')

    train_email_ids_per_sender = preprocess.get_restricted_email_ids_per_sender(
        training, train_mids)

    return train_info, train_email_ids_per_sender, val_info, val_email_ids_per_sender
