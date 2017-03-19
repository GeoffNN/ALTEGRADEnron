from collections import Counter, defaultdict
import operator
import math
import pandas as pd
# from tqdm import tqdm_notebook


def add_time_rank_to_dataframe(df):

    # Set mails with absurd dates to date before any real mail
    min_date_string = '1998-12-20 00:00:01'

    correct_prefix = [chars[0:2] == '00' for chars in list(df['date'])]
    df.loc[correct_prefix, 'date'] = min_date_string

    # Convert dates to timestamps
    df['parsed_date'] = pd.to_datetime(df['date'], yearfirst=True)

    # Add time rank column and sort dataframe
    df['time_rank'] = df['parsed_date'].rank(ascending=False)
    df = df.sort_values('parsed_date')
    return df


def recency_predictions_to_standard(recency_predictions):
    standard_prediction = {}
    for sender, predictions in recency_predictions.items():
        mids = predictions[0]
        if(mids):  # at least one mid for sender
            recipients = predictions[1][0]
            for mid in mids:
                standard_prediction[mid] = recipients
    return standard_prediction


def get_frequency_address_books(emails_ids_per_sender, email_df):
    """
    Create address book with frequency information for each user
    @return address_books with sender as key and recipients as ranked list
    """
    address_books = {}
    idx_sender = 0

    pbar_senders = tqdm_notebook(emails_ids_per_sender.items())
    for sender, ids in pbar_senders:
        recs_temp = []
        for my_id in ids:
            recipients = email_df[email_df['mid'] == int(my_id)][
                'recipients'].tolist()
            recipients = recipients[0].split(' ')
            # keep only legitimate email addresses
            recipients = [rec for rec in recipients if '@' in rec]
            recs_temp.append(recipients)
        # flatten
        recs_temp = [elt for sublist in recs_temp for elt in sublist]
        # compute recipient counts
        rec_occ = dict(Counter(recs_temp))
        # order by frequency
        sorted_rec_occ = sorted(
            rec_occ.items(), key=operator.itemgetter(1), reverse=True)
        # save
        address_books[sender] = sorted_rec_occ

        if idx_sender % 10 == 0:
            print('processed {nb_sender} senders'.format(nb_sender=idx_sender))
        idx_sender += 1
    return address_books


def get_recency_address_books(emails_ids_per_sender, email_df,
                              beta, use_rank=False):
    """
    Create address book with recency information for each user
    @return address_books with sender as key and recipients as ranked list
    """
    address_books = {}

    # get time ranks if @use_rank, else get datetimes
    if(use_rank):
        email_recency_rank_dic = get_recency_rank_dic(email_df)
    else:
        email_recency_datetime_dic = get_rencency_datetime_dic(email_df)
        latest_email_date = email_df.parsed_date.max()

    pbar_senders = tqdm_notebook(emails_ids_per_sender.items())
    for sender, ids in pbar_senders:
        recipients_scores = defaultdict(lambda: 0)
        for my_id in ids:
            my_id = int(my_id)
            if(use_rank):
                # Compute rank scores
                recency_score = get_email_recency_rank_score(
                    email_recency_rank_dic, my_id, beta)
            else:
                # Compute datetime scores
                recency_score = get_email_recency_time_score(
                    email_recency_datetime_dic, my_id, beta, latest_email_date)

            recipients = email_df[email_df['mid'] == int(my_id)][
                'recipients'].tolist()
            recipients = recipients[0].split(' ')
            # keep only legitimate email addresses
            recipients = [rec for rec in recipients if '@' in rec]
            for recipient in recipients:
                recipients_scores[recipient] += recency_score
        # flatten
        rec_occ = dict(Counter(recipients_scores))
        # order by frequency
        sorted_rec_occ = sorted(
            rec_occ.items(), key=operator.itemgetter(1), reverse=True)
        # save
        address_books[sender] = sorted_rec_occ

    return address_books


def predictions_from_addressbook(test_dic, address_books,
                                 keep_all=False, k=10):
    """
    Writes results to csv file for kaggle submission
    for text-independent models (frequency, recency)
    @address_books must be a dict with sender as key and recpient as value
    with recipients ordered according to rank
    @test_dic is a dictionnary with email address as key and mid list as value
    as returned by get_restricted_email_ids_per_sender or
    get_email_ids_per_sender from preprocess.py
    """
    predictions_per_sender = {}
    for sender, mids_predict in test_dic.items():
        recency_preds = []
        if(keep_all):
            k_most = [elt[0] for elt in address_books[sender]]
        else:
            # select k most frequent recipients for the user
            k_most = [elt[0] for elt in address_books[sender][:k]]
        for id_predict in mids_predict:
            # for recency baselines, the predictions are always the same
            recency_preds.append(k_most)
        predictions_per_sender[sender] = [mids_predict, recency_preds]
    return predictions_per_sender


def get_recency_rank_dic(email_df):
    """
    Creates dictionnary that links mid to time rank
    (1 for most recent mail)
    """
    recency_dic = dict(zip(email_df['mid'], email_df['time_rank']))
    return recency_dic


def get_rencency_datetime_dic(email_df):
    recency_dic = dict(zip(email_df['mid'], email_df['parsed_date']))
    return recency_dic


def get_email_recency_rank_score(mid_rank_dic, mid, beta):
    """
    Compute the recency score for a given mid
    """
    mid_rank = mid_rank_dic[mid]
    recency_score = math.exp(- mid_rank / beta)
    return recency_score


def get_email_recency_time_score(mid_datetime_dic, mid, beta, last_email_date):
    """
    Compute the recency score for a given mid
    """
    mid_date_time = mid_datetime_dic[mid]
    recency_score = math.pow(0.5, (last_email_date - mid_date_time
                                   ).total_seconds() / beta)
    return recency_score
