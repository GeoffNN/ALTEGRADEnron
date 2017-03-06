try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x


def get_email_ids_per_sender(email_df):
    """
    returns dictionnary with email address as key and mid list as value
    """
    emails_ids_per_sender = {}
    for index, series in email_df.iterrows():
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
    return recipients


def body_dict_from_panda(dataframe):
    """
    Constructs dictionnary of bodies from dataframe with mid as key
    """
    body_dict = {}
    nb_total = len(dataframe)
    print('Constructing dictionnary from dataframe...')
    pbar_rows = tqdm_notebook(dataframe.iterrows())
    for id, row in pbar_rows:
        body_dict[row.mid] = row.body
    print('done !')
    return body_dict


def get_all_senders(email_ids_per_sender):
    """
    Returns all unique sender names
    """
    return email_ids_per_sender.keys()


def get_all_recipients(address_book):
    """
    @return all_recs a list of all recipients for the given address_book
    with address_books with sender as key and recipients as ranked list
    """
    all_recs = list(
        set([elt[0] for sublist in address_books.values() for elt in sublist]))
    return all_recs
