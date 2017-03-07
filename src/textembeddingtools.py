from collections import defaultdict
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import pickle
import re
import string
from tqdm import tqdm_notebook

import src.preprocess as preprocess


def compute_similarity_scores(model, index_similarities, hdp_dic,
                              idx_to_mids, training_info,
                              test_info, nb_similars=100):

    mid_recipient_scores = {}

    body_dict = preprocess.body_dict_from_panda(test_info)

    pbar_test_bodies = tqdm_notebook(body_dict.items())
    for test_mid, test_body in pbar_test_bodies:
        best_mids, best_scores = get_k_similars(model, index_similarities,
                                                hdp_dic, idx_to_mids,
                                                test_body, k=nb_similars)

        # Get corresponding similarity scores
        test_mail_scores = defaultdict(lambda: 0)
        for train_mid, train_score in zip(best_mids, best_scores):
            recipients = preprocess.get_recipients(training_info, train_mid)
            for recipient in recipients:
                test_mail_scores[recipient] += train_score
        mid_recipient_scores[test_mid] = test_mail_scores
    return mid_recipient_scores


def get_k_similars(model, index_similarities, dictionary,
                   idx_to_mids, email_body, k=100):
    """
    Gets similar indexes for @email_body as a string
    @model and @index_similarities as returned by compute_hdp_model
    @dictionnary as returned by gensim.corpora.Dictionary
    @idx_to_mids {mid_1:idx_1, ...}
    where idx is the corresponding index in index_similarities
    """
    email_tokens = tokenize_body(email_body)
    vec_bow = dictionary.doc2bow(email_tokens)
    query_vec = model[vec_bow]
    similars = index_similarities[query_vec]
    sorted_similars = sorted(enumerate(similars), key=lambda item: -item[1])
    sorted_similars = sorted_similars[:k]
    mids = [idx_to_mids[sim[0]] for sim in sorted_similars]
    scores = [sim[1] for sim in sorted_similars]
    return mids, scores


def get_token_dict(token_dict_path, body_dict, overwrite=False, save=True):
    if (os.path.exists(token_dict_path) and not overwrite):
        with open(token_dict_path, 'rb') as infile:
            token_dict = pickle.load(infile)
    else:
        # Compute token_dict

        token_dict = {}
        pbar_token_dict = tqdm_notebook(body_dict.items())
        for mid, body in pbar_token_dict:
            token_dict[mid] = tokenize_body(body)

        # Save for future use
        if(save):
            with open(token_dict_path, 'wb') as outfile:
                pickle.dump(token_dict, outfile)
    return token_dict


def tokenize_body(body, remove_punctuation=True):
    if(remove_punctuation):
        # Replace punctuation with spaces
        body = re.sub(r'[^\w\s]', ' ', body)
    punctuation_list = list(string.punctuation)
    stop_list = stopwords.words('english') + punctuation_list

    tokens = [symbol for symbol in word_tokenize(
        body.lower()) if symbol not in stop_list]
    return tokens


def compute_hdp_model(email_corpus, word_id_dic,
                      model_results_path, overwrite=False):
    """
    Computes the gensim hdp model utilities
    @model to transform the test bodies to vectors
    @index_similarities to compute the distances to the train corpus
    """
    # Check if model already computed and load in this case
    file_exists_sim = (os.path.exists(model_results_path + 'sim'))
    file_exists_model = (os.path.exists(model_results_path + 'model'))
    file_exists = file_exists_model and file_exists_sim

    if(not overwrite and file_exists):
        with open(model_results_path + 'sim', 'rb') as infile:
            index_similarities = pickle.load(infile)
        with open(model_results_path + 'model', 'rb') as infile:
            model = pickle.load(infile)
    else:
        # Compute model and index for similarities
        print('this will take some time...')
        model = models.HdpModel(email_corpus, id2word=word_id_dic)
        print('computed hdp model')
        index_similarities = similarities.MatrixSimilarity(model[email_corpus])
        print('computed similarity index')

        # Save to files to save time next time
        with open(model_results_path + 'sim', 'wb') as outfile:
            pickle.dump(index_similarities, outfile)
        with open(model_results_path + 'model', 'wb') as outfile:
            pickle.dump(model, outfile)
    return model, index_similarities


def remove_rare_words(email_corpus, threshold_count=1):
    """
    @email_corpus as list of list of words
    [[word_1_1, word_1_2, ...], [word_2_1, word_2_2, ....], ...]
    Removes words that appear less then @thershold_count
    """
    word_counts = defaultdict(int)

    pbar_emails = tqdm_notebook(email_corpus)
    frequency = defaultdict(int)
    for email in pbar_emails:
        for word in email:
            frequency[word] += 1

    email_corpus = [[token for token in text if frequency[token] > threshold_count]
                    for text in email_corpus]
    return email_corpus