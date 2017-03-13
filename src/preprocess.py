try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x

import scipy
from sklearn.preprocessing import MultiLabelBinarizer


def get_email_ids_per_sender(email_df):
    """
    returns dictionnary with email address as key and mid list as value
    """
    emails_ids_per_sender = {}

    row_pbar = tqdm_notebook(email_df.iterrows())
    for index, series in row_pbar:
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        ids = [int(mid) for mid in ids]
        emails_ids_per_sender[sender] = ids
    return emails_ids_per_sender


def get_restricted_email_ids_per_sender(email_df, mids):
    """
    returns dictionnary with email address as key and mid list as value
    mid list is filtered to contain only values present in @mids
    """
    emails_ids_per_sender = {}
    mids = [int(mid) for mid in mids]
    mids = list(mids)

    row_pbar = tqdm_notebook(email_df.iterrows())
    for index, series in row_pbar:
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        ids = [int(mid) for mid in ids]
        ids = [mid for mid in ids if mid in mids]
        emails_ids_per_sender[sender] = ids

    return emails_ids_per_sender


def get_recipients(email_df, mid):
    recipients = email_df[email_df['mid'] == int(mid)]['recipients'].tolist()
    recipients = recipients[0].split(' ')
    recipients = [rec for rec in recipients if '@' in rec]
    return recipients


def body_dict_from_panda(df_info, disp_adv=True):
    """
    Constructs dictionnary of bodies from dataframe with mid as key
    """
    body_dict = {}
    nb_total = len(df_info)
    # print('Constructing dictionnary from dataframe...')

    if(disp_adv):
        row_pbar = tqdm_notebook(df_info.iterrows())
    else:
        row_pbar = df_info.iterrows()
    for id, row in row_pbar:
        body_dict[row.mid] = row.body
    return body_dict


def get_all_senders(df):
    """
    Returns all unique sender names
    """
    return df.sender.values


def get_all_recipients(address_book):
    """
    @return all_recs a list of all recipients for the given address_book
    with address_books with sender as key and recipients as ranked list
    """
    all_recs = list(
        set([elt[0] for sublist in address_book.values() for elt in sublist]))
    return all_recs


def get_conversation_ids(emails_ids_per_sender, df_info):
    """
    :return: dict of dict, with keys: sender then recipient with mids as values
    """
    conversation_ids = {}

    for sender, ids in emails_ids_per_sender.items():
        recs_temp = []
        for my_id in ids:
            recipients = df_info[df_info['mid'] == int(my_id)][
                'recipients'].tolist()
            recipients = recipients[0].split(' ')
            # keep only legitimate email addresses
            recipients = [rec for rec in recipients if '@' in rec]
            recs_temp.append(recipients)
            for rec in recipients:
                if sender in conversation_ids.keys():
                    if rec in conversation_ids[sender].keys():
                        conversation_ids[sender][rec].append(my_id)
                    else:
                        conversation_ids[sender][rec] = [my_id]
                else:
                    conversation_ids[sender] = {rec: [my_id]}
    return conversation_ids


def get_all_recipients_from_df(train_df_info):
    """
    returns list of all recipients in the dataframe
    """
    all_recipients = []

    pbar_rows = tqdm_notebook(train_df_info.iterrows())
    for index, row in pbar_rows:
        recipients = row['recipients'].split(' ')
        recipients = [rec for rec in recipients if '@' in rec]
        all_recipients = recipients + all_recipients
    all_recipients = list(set(all_recipients))
    return all_recipients


def get_mid_sender_dict(emails_ids_per_sender):
    mid_sender_dic = {}
    for sender, mids in emails_ids_per_sender.items():
        for mid in mids:
            mid_sender_dic[mid] = sender
    return mid_sender_dic


def get_sparse_sender_info(idx_to_mids, sender_idx_dic,
                           emails_ids_per_sender, df_info):
    """
    gets sender info as one-hot encoding in sparse matrix
    idx matching mids as in @idx_to_mids in rows and days in columns
    @sender_idx_dic should be the same dic for training and testing
    to preserve sender : idx correspondance
    """

    mid_sender_dic = get_mid_sender_dict(emails_ids_per_sender)
    nb_samples = len(idx_to_mids)
    nb_senders = len(sender_idx_dic)
    sender_features = scipy.sparse.lil_matrix((nb_samples, nb_senders))
    for idx, mid in idx_to_mids.items():
        # Get sender info for given mid
        sender = mid_sender_dic[mid]
        sender_idx = sender_idx_dic[sender]
        # Store as one hot encoding
        sender_features[idx, sender_idx] = 1
    return sender_features


def get_sender_idx_dics(email_ids_per_sender):
    senders = email_ids_per_sender.keys()
    sender_idx_dic = {}
    for idx, sender in enumerate(senders):
        sender_idx_dic[sender] = idx
    return sender_idx_dic


def get_ordered_recipients(email_info, idx_to_mids):
    nb_mids = len(idx_to_mids)
    recipients = [[]] * nb_mids
    pbar_mids = tqdm_notebook(idx_to_mids.items())
    for idx, mid in pbar_mids:
        mid_recipients = get_recipients(email_info, mid)
        recipients[idx] = mid_recipients
    return recipients


def get_ordered_recipients(email_info, idx_to_mids):
    nb_mids = len(idx_to_mids)
    recipients = [[]] * nb_mids
    pbar_mids = tqdm_notebook(idx_to_mids.items(), leave=False)
    for idx, mid in pbar_mids:
        mid_recipients = get_recipients(email_info, mid)
        recipients[idx] = mid_recipients
    return recipients


def get_one_hot_sender_recipients(sender_idx_to_mids, email_info, disp=True):
    """
    @param sender_idx_to_mids : {sender:{0:mid_1, 1:mid_2, ...}, ...}
    @return sender_recipients_binaries : {sender: binarized_recipients}
    @return sender_idx_to_recipients : {sender: {0:recipient_1, ...},...}
    sender_idx_to_recipients codes for the correspondance between
    recipients and the rows in binarized_recipients
    """

    sender_recipients_binaries = {}
    sender_recipients_non_binary = {}
    sender_idx_to_recipients = {}
    if(disp):
        pbar_senders = tqdm_notebook(sender_idx_to_mids.items())
    else:
        pbar_senders = sender_idx_to_mids.items()
    for sender, idx_to_mids in pbar_senders:
        ordered_recipients = get_ordered_recipients(email_info, idx_to_mids)
        binarizer = MultiLabelBinarizer()
        binarized_recipients = binarizer.fit_transform(ordered_recipients)
        sender_recipients_binaries[sender] = binarized_recipients
        sender_recipients_non_binary[sender] = ordered_recipients

        all_recipients = binarizer.classes_
        idx_to_recipients = dict(
            zip(range(len(all_recipients)), all_recipients))
        sender_idx_to_recipients[sender] = idx_to_recipients

    return sender_recipients_binaries, sender_idx_to_recipients
