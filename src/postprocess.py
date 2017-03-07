def write_recency_results(predictions_per_mid,
                          path_to_results, results_name):
    """
    Writes results to csv file for kaggle submission
    result must be a dict {sender:[[mids,[senders, senders, ...]]}
    """
    with open(path_to_results + results_name, 'wb') as my_file:
        my_file.write(bytes('mid,recipients\n', 'UTF-8'))
        for sender, preds in predictions_per_mid.items():
            ids = preds[0]
            freq_preds = preds[1]
            for index, my_preds in enumerate(freq_preds):
                my_file.write(bytes(str(ids[index]) + ',' +
                                    ' '.join(my_preds) + '\n', 'UTF-8'))


def write_results_ranked(predictions_per_mid,
                         path_to_results, results_name):
    """
    Writes results to csv file for kaggle submission
    result must be a dict {mid:[recipient1, recipient2, ...]}
    """
    with open(path_to_results + results_name, 'wb') as my_file:
        my_file.write(bytes('mid,recipients\n', 'UTF-8'))
        for mid, preds in predictions_per_mid.items():
            my_file.write(bytes(str(mid) + ',' +
                                ' '.join(preds) + '\n', 'UTF-8'))


def write_results_probas(results, path_to_results, results_name):
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
