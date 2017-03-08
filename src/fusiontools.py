from collections import defaultdict
import math
import operator


def reciprocal_rank_score(rank, constant, ranking='linear'):
    if (ranking == 'linear'):
        score = 1 / (constant + rank)
    elif(ranking == 'exponential'):
        score = math.exp(-rank / constant)
    else:
        raise ValueError("Didn't receive 'linear' \
                         or 'exponential' as ranking param")
    return score


def reciprocal_rerank(models, ranking_constant,
                      nb_recipients=10, weights=None):
    fusion_dic = {}
    for mid in models[0]:
        all_recipients = [
            recipient for model in models for recipient in model[mid]]
        unique_recipients = list(set(all_recipients))
        mid_recipient_ranks = defaultdict(lambda: 0)
        for recipient in unique_recipients:
            for idx_model, model in enumerate(models):
                if(weights):
                    weight = weights[idx_model]
                else:
                    weight = 1
                # Add model reranked score
                model_recipients = model[mid]
                if (recipient in model_recipients):
                    rank = model[mid].index(recipient)

                    # Add recipient rank score
                    rank_score = reciprocal_rank_score(rank, ranking_constant)
                    mid_recipient_ranks[recipient] += rank_score * weight
        sorted_recipient_ranks = sorted(
            mid_recipient_ranks.items(),
            key=operator.itemgetter(1), reverse=True)
        # sorted_recipient_ranks is in [(sender1, score1), (sender2, score2),
        # ...] format
        reranked_recipients = [recipient[0]
                               for recipient in sorted_recipient_ranks]
        # Keep only top predictions
        fusion_dic[mid] = reranked_recipients[:nb_recipients]
    return fusion_dic


def reciprocal_rerank2(models, ranking_constant,
                       nb_recipients=10, weights=None):
    fusion_dic = {}
    for mid in models[0]:
        all_recipients = [
            recipient for model in models for recipient in model[mid]]
        unique_recipients = list(set(all_recipients))
        mid_recipient_ranks = defaultdict(lambda: 0)
        for recipient in unique_recipients:
            for idx_model, model in enumerate(models):
                if(weights):
                    weight = weights[idx_model]
                else:
                    weight = 1
                # Add model reranked score
                model_recipients = model[mid]
                if (recipient in model_recipients):
                    rank = model[mid].index(recipient)

                    # Add recipient rank score
                    rank_score = reciprocal_rank_score(
                        rank, weight * ranking_constant)
                    mid_recipient_ranks[recipient] += rank_score
        sorted_recipient_ranks = sorted(
            mid_recipient_ranks.items(),
            key=operator.itemgetter(1), reverse=True)
        # sorted_recipient_ranks is in [(sender1, score1), (sender2, score2),
        # ...] format
        reranked_recipients = [recipient[0]
                               for recipient in sorted_recipient_ranks]
        # Keep only top predictions
        fusion_dic[mid] = reranked_recipients[:nb_recipients]
    return fusion_dic


def keep_only_max_recips(dic_ranks, max_recips=10):
    """
    Keeps only top @max_recips for each mid in dic_ranks
    Input and output dic in same format : {mid:[sender1, sender2, ...], }
    Also converts mid key from str to int if needed !
    """
    cropped_recips = {}
    for mid, recipients in dic_ranks.items():
        cropped_recips[int(mid)] = list(recipients[:max_recips])
    return cropped_recips
