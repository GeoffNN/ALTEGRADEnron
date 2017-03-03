import pandas as pd
import numpy as np
from numpy.linalg import norm

from src.graph import get_neighbors


# Intuitive pipeline
def get_recommandation(mail_body, mail_date, sender, graph):
    global conversation_ids
    recs = get_neighbors(graph, sender)
    sims = pd.Series([])
    for rec in recs:
        c = centroid(conversation_ids[(sender, rec)])
        sims[rec] = similarity(mail_body, c)
    sims.sort(inplace=True)
    return sims[:10].index


def centroid(mails_ids):
    # TODO: compute centroid of the given email list
    pass


def similarity(f_mail1, f_mail2):
    return np.dot(f_mail1, f_mail2) / (norm(f_mail1) * norm(f_mail2))
