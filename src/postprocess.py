def write_results(results, path_to_results, results_name):
    """
    Writes results to csv file for kaggle submission
    result must be a dict {mid:[(prob1, sender1), ...]}
    """
    with open(path_to_results + results_name, 'wb') as f:
        f.write(bytes('mid,recipients\n', 'UTF-8'))
        for mid, preds in results.items():
            preds_no_prob = [x[1] for x in preds]
            f.write(bytes(str(mid) + ',' +
                          ' '.join(preds_no_prob) + '\n', 'UTF-8'))


def write_prediction_from_addressbook(test_df, address_books, path_to_results,
                                      result_name, k=10):
    """
    Writes results to csv file for kaggle submission
    for text-independent models (frequency, recency)
    address_books must be a dict with sender as key and recpient as value
    with recipients ordered according to rank
    """
    predictions_per_sender = {}
    for index, row in test_df.iterrows():
        name_ids = row.tolist()
        sender = name_ids[0]
        # get IDs of the emails for which recipient prediction is needed
        mids_predict = name_ids[1].split(' ')
        mids_predict = [int(my_id) for my_id in mids_predict]
        recency_preds = []
        # select k most frequent recipients for the user
        k_most = [elt[0] for elt in address_books[sender][:k]]
        for id_predict in mids_predict:
            # for recency baselines, the predictions are always the same
            recency_preds.append(k_most)
        predictions_per_sender[sender] = [mids_predict, recency_preds]
    with open(path_to_results + result_name, 'wb') as my_file:
        my_file.write(bytes('mid,recipients\n', 'UTF-8'))
        for sender, preds in predictions_per_sender.items():
            ids = preds[0]
            freq_preds = preds[1]
            for index, my_preds in enumerate(freq_preds):
                my_file.write(bytes(str(ids[index]) + ',' +
                                    ' '.join(my_preds) + '\n', 'UTF-8'))
