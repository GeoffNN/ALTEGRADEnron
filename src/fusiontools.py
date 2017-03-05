from collections import defaultdict
import math
import operator


def reciprocal_rank_score(rank, constant):
    return 1 / (constant + rank)
    # return math.exp(-rank / constant)


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
