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
    sender_counter = 0
    for index, series in email_df.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        ids = [int(mid) for mid in ids]
        ids = [mid for mid in ids if mid in mids]
        emails_ids_per_sender[sender] = ids

        # Display advancement
        if(sender_counter % 20 == 0):
            print('processed {sender_nb} senders'.format(
                sender_nb=sender_counter))
        sender_counter += 1
    return emails_ids_per_sender


def get_recipients(email_df, mid):
    recipients = email_df[email_df['mid'] == int(mid)]['recipients'].tolist()
    recipients = recipients[0].split(' ')
    return recipients


def body_dict_from_panda(df_info):
    """
    Constructs dictionnary of bodies from dataframe with mid as key
    """
    body_dict = {}
    nb_total = len(df_info)
    print('Constructing dictionnary from dataframe...')
    for id, row in df_info.iterrows():
        body_dict[row.mid] = row.body
        if (id % 10000 == 0):
            print('{id} / {nb_total}'.format(id=id, nb_total=nb_total))
    print('done !')
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
