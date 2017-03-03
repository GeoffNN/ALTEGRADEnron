def get_email_ids_per_sender(email_df):
    """
    returns dictionnary with email address as key and mid list as value
    """
    emails_ids_per_sender = {}
    for index, series in email_df.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        emails_ids_per_sender[sender] = ids
    return emails_ids_per_sender


def body_dict_from_panda(dataframe):
    """
    Constructs dictionnary of bodies from dataframe with mid as key
    """
    body_dict = {}
    nb_total = len(dataframe)
    print('Constructing dictionnary from dataframe...')
    for id, row in dataframe.iterrows():
        body_dict[row.mid] = row.body
        if(id % 10000 == 0):
            print('{id} / {nb_total}'.format(id=id, nb_total=nb_total))
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
