import pandas as pd

def build_graph(address_book):
    adj_by_sender = {}
    for sender, val in address_book.items():
        adj_by_sender[sender] = pd.Series([el[1] for el in val], index=[el[0] for el in val], name=sender)
    adjacency_mat = pd.DataFrame.from_dict(adj_by_sender).T.fillna(0)
    adjacency_mat = adjacency_mat.T.dot(adjacency_mat)
    return adjacency_mat

def get_neighbors(sender, adjacency_matrix):
    return adjacency_matrix.loc[sender].nonzero()[0]