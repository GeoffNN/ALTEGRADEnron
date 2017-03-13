import copy
import igraph
import itertools
import numpy as np
import scipy
from sklearn.preprocessing import normalize
try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x


import src.tfidftools as tfidftools


def terms_to_graph(terms, window_size=4):
    # This function returns a directed weighted igraph
    # from a list of terms (the tokens from the pre-processed text)
    # e.g., ['quick','brown','fox']
    # Edges are weighted based on term co-occurence within a sliding window of
    # fixed size 'w'

    from_to = {}

    # create empty graph
    graph = igraph.Graph(directed=True)

    # Ensure email larger then at least sliding window
    if(len(terms) >= window_size):
        # Create initial complete graph (first window_size terms)
        terms_temp = terms[0:window_size]
        term_indexes = list(itertools.combinations(range(window_size), r=2))
        new_edges = []

        for index_tuple in term_indexes:
            new_edges.append(tuple([terms_temp[i] for i in index_tuple]))

        for new_edge in new_edges:
            if new_edge in from_to:
                from_to[new_edge] += 1
            else:
                from_to[new_edge] = 1

        # then iterate over the remaining terms
        for i in range(window_size, len(terms)):
            # term to consider
            considered_term = terms[i]
            # all terms within sliding window
            terms_temp = terms[(i - window_size + 1):(i + 1)]

            # edges to try
            candidate_edges = []
            for p in range(window_size - 1):
                candidate_edges.append((terms_temp[p], considered_term))

            for try_edge in candidate_edges:

                # if not self-edge
                if try_edge[1] != try_edge[0]:

                    # if edge has already been seen, update its weight
                    if try_edge in from_to:
                        from_to[try_edge] += 1

                    # if edge has never been seen, create it and assign it a unit
                    # weight
                    else:
                        from_to[try_edge] = 1

        # add vertices
        graph.add_vertices(sorted(set(terms)))

        # add edges, direction is preserved since the graph is directed
        graph.add_edges(from_to.keys())

        # set edge and vertice weights
        # based on co-occurence within sliding window
        graph.es['weight'] = list(from_to.values())
        graph.vs['weight'] = graph.strength(
            weights=list(from_to.values()))  # weighted degree

    return(graph)


def get_term_indegree(graph):
    """"
    Gets dictionnary of words and indegrees for the @graph
    returns empty dictionnary in empty graph
    """
    # work on clone of g to preserve g
    graph_copy = copy.deepcopy(graph)
    indegree_dic = {}
    indegrees = graph_copy.indegree()
    if(indegrees):
        terms = graph.vs['name']
    else:
        terms = []
    return dict(zip(terms, indegrees))


def unweighted_k_core(graph):
    # work on clone of g to preserve g
    graph_copy = copy.deepcopy(graph)

    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(graph_copy.vs['name'], [0] * len(graph_copy.vs)))

    i = 0

    # while there are vertices remaining in the graph
    while len(graph_copy.vs) > 0:
        # while there is a vertex with degree less than i
        while [deg for deg in graph_copy.strength() if deg <= i]:
            index = [ind for ind, deg in enumerate(
                graph_copy.strength()) if deg <= i][0]
            # assign i to the vertices core numbers
            cores_g[graph_copy.vs[index]['name']] = i
            graph_copy.delete_vertices(index)
        print(i)
        i += 1

    return cores_g


def get_twidf_matrix(token_dict, doc_lengths_dic, average_doc_len, idf_dic,
                     idf_words, tw_constant=0.003):
    """
    Computes tw_idf model in form of tw_idf_matrix
    with samples in rows and features in columns
    """
    # Create graph dict
    graph_dict = get_graph_dict(token_dict)

    # Great matrix representation as rows of features
    tw_vectors = scipy.sparse.lil_matrix(((len(graph_dict), len(idf_words))))
    mids = list(graph_dict.keys())
    # Create idx-mid correspondance
    twidf_mids = dict(zip(range(len(mids)), mids))
    pbar_mids = tqdm_notebook(twidf_mids.items())
    for idx, mid in pbar_mids:
        doc_length = doc_lengths_dic[mid]
        email_graph = graph_dict[mid]
        indegree_weights_dic = get_term_indegree(email_graph)
        vector = get_twidf_vector_from_indegree(indegree_weights_dic, idf_dic,
                                                idf_words, doc_length,
                                                average_doc_len, tw_constant)
        tw_vectors[idx, :] = vector
    tw_vectors = scipy.sparse.csr_matrix(tw_vectors)
    return tw_vectors, twidf_mids


def get_twidf_vector_from_indegree(indegree_weights_dic, idf_dic,
                                   idf_words, doc_length, average_doc_length,
                                   tw_constant=0.003):
    vec = np.zeros(len(idf_words))
    for word, indegree in indegree_weights_dic.items():
        if word in idf_words:
            idx = idf_words.index(word)
            tw_term = indegree / \
                (1 - tw_constant + tw_constant * (doc_length / average_doc_length))
            vec[idx] = tw_term * idf_dic[word]
    vec = vec.reshape(1, -1)
    vec = normalize(vec)
    return vec


def get_graph_dict(token_dict):
    graph_dict = {}
    pbar_token_dict = tqdm_notebook(token_dict.items())
    for mid, tokens in pbar_token_dict:
        current_graph = terms_to_graph(tokens, window_size=4)
        graph_dict[mid] = current_graph
    return graph_dict


def get_twidf_vectors_from_tokens(idf_dic, idf_words, token_dict,
                                  average_doc_len):
    """
    @idf_dic and @idf_words as returned by tfidftools.get_idf_dic
    """
    vectors = {}
    pbar_tokens = tqdm_notebook(token_dict.items())
    for mid, tokens in pbar_tokens:
        # Compute query vector
        query_vector = get_twidf_vector_from_tokens(idf_dic, idf_words,
                                                    tokens, average_doc_len)
        vectors[mid] = query_vector
    return vectors


def get_twidf_vector_from_tokens(idf_dic, idf_words, tokens_list,
                                 average_doc_len):
    doc_length = len(tokens_list)
    graph = terms_to_graph(tokens_list, window_size=4)
    indegrees = get_term_indegree(graph)
    query_vector = get_twidf_vector_from_indegree(indegrees, idf_dic,
                                                  idf_words,
                                                  doc_length,
                                                  average_doc_len)
    return query_vector
