from collections import defaultdict
import operator
from sklearn.metrics.pairwise import linear_kernel
try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x

import src.tfidftools as tfidftools
import src.preprocess as preprocess


def compute_similarity_scores(tfidf_model, tfidf_matrix,
                              tfidf_mids, training_info,
                              test_info, nb_similars=100):
    """
    Computes similarity scores for mails in @test_info dataframe

    @params tfidf_model, tfidf_matrix, tfidf_mids
    as returned by tfidftools.get_tfidf

    @return mid_recipient_scores
    dic {mid:{recipient:sum_of_si,ilarities, ...}, ...}
    """
    test_mids = list(test_info['mid'])
    mid_recipient_scores = {}

    test_mids_pbar = tqdm_notebook(test_mids)
    for test_mid in test_mids_pbar:
        query_vector = tfidftools.get_tfidf_vector(
            test_mid, test_info, tfidf_model)
        similars = find_similar(query_vector, tfidf_matrix,
                                nb_similars=nb_similars)

        # Get mid in training set corresponding to best matches
        best_match_mid = [tfidf_mids[similar_item[0]]
                          for similar_item in similars]

        # Get corresponding similarity scores
        best_match_scores = [similar_item[1] for similar_item in similars]
        test_mail_scores = defaultdict(lambda: 0)
        for train_mid, train_score in zip(best_match_mid, best_match_scores):
            recipients = preprocess.get_recipients(training_info, train_mid)
            for recipient in recipients:
                test_mail_scores[recipient] += train_score
        mid_recipient_scores[test_mid] = test_mail_scores
    return mid_recipient_scores


def find_similar(vector, tfidf_matrix, nb_similars=100):
    # ref
    # http://www.markhneedham.com/blog/2016/07/27/scitkit-learn-tfidf-and-cosine-similarity-for-computer-science-papers/
    cosine_similarities = linear_kernel(vector, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    indexes_similarities = [(index, cosine_similarities[index])
                            for index in related_docs_indices][0:nb_similars]
    return indexes_similarities


def similar_dic_to_standard(indexes_similarities,
                            keep_all=False, nb_recipients=10):
    knn_dic = {}
    for mid in indexes_similarities:
        receiver_scores = indexes_similarities[mid]
        sorted_cos_sim = sorted(receiver_scores.items(),
                                key=operator.itemgetter(1), reverse=True)
        if (keep_all):
            k_most = [elt[0] for elt in sorted_cos_sim]
        else:
            k_most = [elt[0] for elt in sorted_cos_sim[:nb_recipients]]

        knn_dic[mid] = k_most
    return knn_dic
