from collections import defaultdict
from gensim import corpora, models, similarities
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import pickle
import scipy
import re
import string
from tqdm import tqdm_notebook
from sklearn.ensemble import RandomForestClassifier
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
    @model and @index_similarities as returned by compute_model
    and compute_sim_matrix
    @dictionnary as returned by gensim.corpora.Dictionary
    @idx_to_mids {mid_1:idx_1, ...} for the mids that match index_similarities
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


def get_token_dict(body_dict, save=False, use_saved=False,
                   token_dict_path='', disp_adv=True):
    if (os.path.exists(token_dict_path) and use_saved):
        with open(token_dict_path, 'rb') as infile:
            token_dict = pickle.load(infile)
    else:
        # Compute token_dict

        token_dict = {}

        if(disp_adv):
            row_pbar = tqdm_notebook(body_dict.items())
        else:
            row_pbar = body_dict.items()

        for mid, body in row_pbar:
            token_dict[mid] = tokenize_body(body)

        # Save for future use
        if(save):
            with open(token_dict_path, 'wb') as outfile:
                pickle.dump(token_dict, outfile)
    return token_dict


def tokenize_body(body, remove_punctuation=True, stemming=False):
    # Initial string preprocessing
    body = body.lower()
    # Remove special characters
    body = body.strip().encode('ascii', 'ignore').decode('ascii')
    if(remove_punctuation):
        # Replace punctuation except '-' with spaces
        body = re.sub(r'[.,\/#!$%\^&\*;:{}=\_`~()]', ' ', body)

    # punct = string.punctuation.replace('-', '')
    # punctuation_list = list(punct)
    stop_list = stopwords.words('english')

    tokens = [symbol for symbol in word_tokenize(
        body) if symbol not in stop_list]
    if(stemming):
        stemmer = PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    return tokens


def token_dicts_to_token_lists(token_dict, rarity_threshold=2):
    """
    Extract @idx_to_mids {idx:mid} and
    @email_list : tokens as list of tokens [[token1_1, token_1_2,...], ...]
    with indexes correspondig to @idx_to_mids
    """
    mids = list(token_dict.keys())
    email_list = list(token_dict.values())
    email_list = remove_rare_words(email_list,
                                   threshold_count=rarity_threshold)

    idx_to_mids = dict(zip(range(len(mids)), mids))
    return email_list, idx_to_mids


def get_sender_model_features_from_tokens(email_ids_per_sender,
                                          token_dict, word_id_dic,
                                          model, feature_size):
    """
    creates a dictionnary {sender: {'idx_to_mids': idx_to_mids},
                            {'features': features}}
    from the given @param token_dict where the row idx features match
    the mids in the idx_to_mids dic
    @param word_id_dic is used to create the corpus list
    from the email list of lists
    @param token_dict : {mid_1: [token_1_1, token_1_2, ...], ....}
    @param feature_size is used to prealocate the
    right size in model_vectors_to_matrix_features
    @param model : trained language model
    """
    senders_mid_features_dic = {}
    senders_idx_to_mid_dic = {}
    pbar_senders = tqdm_notebook(email_ids_per_sender.items())
    for sender, mids in pbar_senders:
        sender_token_dict = {mid: tokens for mid, tokens in token_dict.items()
                             if mid in mids}
        sender_email_list, sender_idx_to_mids = token_dicts_to_token_lists(
            sender_token_dict, rarity_threshold=4)
        sender_email_corpus = [word_id_dic.doc2bow(
            text) for text in sender_email_list]
        sender_model_vectors = model[sender_email_corpus]
        sender_model_matrix = model_vectors_to_matrix_features(
            sender_model_vectors, feature_size)
        senders_mid_features_dic[sender] = sender_model_matrix
        senders_idx_to_mid_dic[sender] = sender_idx_to_mids
    return senders_mid_features_dic, senders_idx_to_mid_dic


def compute_model(email_corpus, word_id_dic, model='lda',
                  nb_topics=200, save=True,
                  model_results_path='', use_saved=False):
    """
    Computes the gensim model utilities
    @model to transform the test bodies to vectors
    @index_similarities to compute the distances to the train corpus
    """
    # Check if model already computed and load in this case
    file_exists = (os.path.exists(model_results_path))

    if(use_saved and file_exists):
        with open(model_results_path, 'rb') as infile:
            index_similarities = pickle.load(infile)
        with open(model_results_path, 'rb') as infile:
            model = pickle.load(infile)
    else:
        # Compute model
        print('this will take some time...')
        if(model == 'lda'):
            model = models.LdaModel(email_corpus, id2word=word_id_dic,
                                    num_topics=nb_topics)
        elif(model == 'hdp'):
            model = models.HdpModel(email_corpus, id2word=word_id_dic)
        elif(model == 'lsi'):
            tfidf = models.TfidfModel(email_corpus)
            email_corpus_tfidf = tfidf[email_corpus]
            model = models.LsiModel(email_corpus_tfidf, id2word=word_id_dic,
                                    num_topics=nb_topics)
        else:
            raise ValueError(
                '{model} is not a valid model name'.format(model=model))
        print('computed model')
        if(save):
            print('saving new model to {path}'.format(
                path=(model_results_path)))
            # Save to files to save time next time
            with open(model_results_path, 'wb') as outfile:
                pickle.dump(model, outfile)
    return model


def model_vectors_to_matrix_features(model_vectors, nb_topics):
    """
    @param nb_topics the number of features in the model
    @param modeL_vectors obtained by applying the model to a corpus
    @return model_features a sparse matrix with topic weights
    in rows for each email
    """

    nb_emails = len(model_vectors)
    model_features = scipy.sparse.lil_matrix((nb_emails, nb_topics))
    pbar_model_vectors = tqdm_notebook(enumerate(model_vectors), leave=False)
    for doc_idx, vec in pbar_model_vectors:
        for topic_index, topic_weight in vec:
            model_features[doc_idx, topic_index] = topic_weight
    model_features = scipy.sparse.csr_matrix(model_features)
    return model_features


def compute_sim_matrix(model, email_corpus, sim_results_path='',
                       use_saved=False, save=False):
    file_exists = (os.path.exists(sim_results_path + 'sim'))
    if(use_saved and file_exists):
        with open(model_results_path + 'sim', 'rb') as infile:
            index_similarities = pickle.load(infile)
    else:
        #  Compute index for similarities
        index_similarities = similarities.MatrixSimilarity(model[email_corpus])
        print('computed similarity index')
        if(save):
            with open(model_results_path + 'sim', 'wb') as outfile:
                pickle.dump(index_similarities, outfile)
    return index_similarities


def remove_rare_words(email_corpus, threshold_count=1):
    """
    @email_corpus as list of list of words
    [[word_1_1, word_1_2, ...], [word_2_1, word_2_2, ....], ...]
    Removes words that appear less then @thershold_count
    """
    word_counts = defaultdict(int)

    frequency = defaultdict(int)
    for email in email_corpus:
        for word in email:
            frequency[word] += 1

    email_corpus = [[token for token in text if frequency[token] > threshold_count]
                    for text in email_corpus]
    return email_corpus


def get_doc_length_info(token_dict):
    doc_lengths_dic = {}
    # Store email token lengths
    for mid, tokens in token_dict.items():
        doc_lengths_dic[mid] = len(tokens)
    # Compute mean doc length
    average_doc_len = sum(doc_lengths_dic.values()) / len(doc_lengths_dic)
    return doc_lengths_dic, average_doc_len


def create_stacked_feature_dic(feature_dics):
    """
    takes a list of dics {sender: sparse_feature_dic} as input as returned by
    recency.get_sender_sparse_date_info or get_sender_model_features_from_tokens
    with matching row indexes
    @return feature_dic {sender: features}
    where features are horizontal stacks of the original features
    """
    final_feature_dic = {}
    # For each sender hstack all features
    for sender in feature_dics[0].keys():
        feature_list = []
        for feature_dic in feature_dics:
            feature_list.append(feature_dic[sender])
            final_feature = scipy.sparse.hstack(feature_list)
        final_feature_dic[sender] = final_feature
    return final_feature_dic


def tree_train_predict(train_stacked_features_dict, train_recipient_binary_dict,
                       train_sender_idx_to_recipients, val_stacked_features_dict,
                       val_senders_idx_to_mid_dic, disp=True, nb_tree=10):
    """
    @param train_sender_idx_to_recipients : {column_idx: recipient, ...}
    """
    predictions = {}
    # Prepare sender loop
    if(disp):
        sender_pbar = tqdm_notebook(val_stacked_features_dict.keys())
    else:
        sender_pbar = val_stacked_features_dict.keys()
    for sender in sender_pbar:
        # Extract sender-specific info
        stacked_train_features = train_stacked_features_dict[sender]
        stacked_val_features = val_stacked_features_dict[sender]
        binarized_train_predictions = train_recipient_binary_dict[sender]
        idx_to_recipients = train_sender_idx_to_recipients[sender]
        idx_to_mid = val_senders_idx_to_mid_dic[sender]
        # !! Make sure that there is more then one potential recipient
        # !! Otherwise prediction format is different
        if(len(idx_to_recipients) == 1):
            only_recipient = list(idx_to_recipients.values())
            for idx, mid in idx_to_mid.items():
                predictions[mid] = only_recipient
            continue

            # Check that at least one feature present for sender
        if(stacked_val_features.shape[0]):
            # Train classifier and predict
            rdm_train = RandomForestClassifier(n_estimators=nb_tree)
            rdm_train.fit(stacked_train_features, binarized_train_predictions)
            binary_preds = rdm_train.predict_proba(
                stacked_val_features)

            # extract class 1 probas in columns
            binary_preds = np.asarray(binary_preds)
            binary_preds = np.rollaxis(binary_preds[:, :, 1], axis=1)

            # order indexes according to decreasing proba
            indexes = binary_preds.argsort(axis=1)[:, ::-1]
            for idx_sample, idx_recipient in enumerate(indexes):
                mid = idx_to_mid[idx_sample]
                recipient_probas = binary_preds[idx_sample, :]
                positive_probas_nb = np.count_nonzero(recipient_probas)
                cropped_indexes = idx_recipient[:positive_probas_nb]
                recipients = [idx_to_recipients[idx]
                              for idx in cropped_indexes]
                predictions[mid] = recipients
    return predictions


def get_predictions_from_binarized(sender_predictions_dic, sender_idx_to_mids_dic,
                                   sender_idx_to_recipients):
    for sender, binary_predictions in sender_predictions_dic:
        pass
