try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x

from collections import defaultdict
from optparse import OptionParser
import os.path
import re


import src.preprocess as preprocess
import src.knntools as knntools


def get_emails_from_text(email_body):
    """Returns an iterator of matched emails found in string s."""
    # Removing lines that start with '//' because the regular expression
    # mistakenly matches patterns like 'http://foo@bar.com' as '//foo@bar.com'.
    regex = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                        "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
                        "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))

    return (email[0].lower() for email in re.findall(regex, email_body)
            if not email[0].startswith('//'))


def get_filtered_emails_dic(body_dict, mid_sender_dic, candidate_list=None):
    emails_in_content = {}
    pbar_emails = tqdm_notebook(body_dict.items())
    for mid, body in pbar_emails:
        sender = mid_sender_dic[mid]
        found_emails = list(get_emails_from_text(body))
        unique_found_emails = list(set(found_emails))
        # Keep only known recipients if list is provided
        if(candidate_list):
            unique_found_emails = [mail.lower()
                                   for mail in unique_found_emails
                                   if mail.lower() in candidate_list]
        # Remove sender email
        filtered_email = [
            mail for mail in unique_found_emails if mail != sender]
        if (filtered_email):
            emails_in_content[mid] = filtered_email
    return emails_in_content


def extract_after_key_word(text, keyword='Subject:', extracted_length=25):
    keyword_begin_index = text.find(keyword)
    found_word = False
    extracted = ''
    if(keyword_begin_index > -1):
        found_word = True
        extracted_begin_index = keyword_begin_index + len(keyword)
        extracted = text[extracted_begin_index: extracted_begin_index +
                         extracted_length]
    return found_word, extracted


def get_keyword_dic(body_dict, keyword, extracted_length=25):
    keyword_dic = {}
    for mid, body in body_dict.items():
        has_subject, subject = extract_after_key_word(body, keyword=keyword,
                                                      extracted_length=extracted_length)
        if(has_subject):
            keyword_dic[mid] = subject
    return keyword_dic


def get_keyword_prediction(train_body_dict, test_body_dict,
                           train_info, keyword, extracted_length=20):
    """
    Creates predictions only for emails that do contain @param keyword
    in the body
    For those that do, predicts same recipients as the emails with same
    @param keyword in their body, with order of recipients according to
    count of recipient presence in those emails
    """
    test_subject_dic = get_keyword_dic(test_body_dict, keyword=keyword,
                                       extracted_length=extracted_length)
    train_subject_dic = get_keyword_dic(train_body_dict, keyword=keyword,
                                        extracted_length=extracted_length)
    keyword_prediction_scores = {}
    pbar_test = tqdm_notebook(test_body_dict.items())
    for mid, test_body in pbar_test:
        recipient_scores = defaultdict(int)
        if (mid in test_subject_dic.keys()):
            test_keyword_query = test_subject_dic[mid]
            mids = [mid for mid, train_subject in train_subject_dic.items()
                    if train_subject == test_keyword_query]
            for hit_mid in mids:
                recipients = preprocess.get_recipients(train_info, hit_mid)
                for recipient in recipients:
                    recipient_scores[recipient] += 1
            keyword_prediction_scores[mid] = recipient_scores
    keyword_predictions = knntools.similar_dic_to_standard(
        keyword_prediction_scores)
    return keyword_predictions
