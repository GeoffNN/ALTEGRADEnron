import src.tools as tools
try: from tqdm import tqdm
except ImportError: tqdm = lambda x:x

def compute_recipient_prior(df_info):
    all_recipients = df_info.recipients.values
    res = {}
    pbar = tqdm(all_recipients)
    # get counts
    for recipients_string in pbar:
        recipients_list = recipients_string.split()
        for recipient in recipients_list:
            res[recipient] = res.get(recipient, 0) + 1
    # normalize to obtain probabilities
    total = sum(res.values())
    for recipient in res:
        res[recipient] /= total
    # retun result
    return res 

def compute_sender_likelihood_given_recipient(df, df_info):
    res = {}
    pbar = tqdm(range(len(df)))
    # get counts
    for i in pbar:
        sender = df.loc[i, 'sender']
        mids = df.loc[i, 'mids'].split()
        for mid in mids:
            recipients = list(df_info.loc[df_info['mid'] == int(mid), 'recipients'])[0].split()
            for recipient in recipients:
                res[recipient] = res.get(recipient, {})
                res[recipient][sender] = res[recipient].get(sender, 0) + 1
    # normalize to obtain probabilities
    for recipient in res:
        total = sum(res[recipient].values())
        for sender in res[recipient]:
            res[recipient][sender] /= total
    # return res
    return res

def compute_mail_likelihood_given_recipient_and_sender(df, df_info):
    res_1 = {}
    res_2 = {}
    res_3 = {}
    pbar = tqdm(range(len(df)))
    # get counts
    for i in pbar:
        sender = df.loc[i, 'sender']
        mids = df.loc[i, 'mids'].split()
        for mid in mids:
            recipients = list(df_info.loc[df_info['mid'] == int(mid), 'recipients'])[0].split()
            body = list(df_info.loc[df_info['mid'] == int(mid), 'body'])[0]
            words = tools.get_tokens(body)
            for recipient in recipients:
                for word in words:
                    # P(w)
                    res_1[word] = res_1.get(word, 0) + 1
                    # P(w| R)
                    res_2[recipient] = res_2.get(recipient, {})
                    res_2[recipient][word] = res_2[recipient].get(word, 0) + 1
                    # P(w|R, S)
                    res_3[recipient] = res_3.get(recipient, {})
                    res_3[recipient][sender] = res_3[recipient].get(sender, {})
                    res_3[recipient][sender][word] = res_3[recipient][sender].get(word, 0) + 1
    # get P(w), normalize to obtain probabilities
    total = sum(res_1.values())
    for word in res_1:
        res_1[word] /= total
    # get P(w|R), normalize to obtain probabilities
    for recipient in res_2:
        total = sum(res_2[recipient].values())
        for word in res_2[recipient]:
            res_2[recipient][word] /= total
    # get P(w|R, S), normalize to obtain probabilities
    for recipient in res_3:
        for sender in res_3[recipient]:
            total = sum(res_3[recipient][sender].values())
            for word in res_3[recipient][sender]:
                res_3[recipient][sender][word] /= total

    return res_1, res_2, res_3

def predict(recipient, sender, email, probs, a=1/3, b=1/3, c=1/3):
    # unpack probs
    p_r = probs['p_r']
    p_s_r = probs['p_s_r']
    p_w_r_s = probs['p_w_r_s']
    p_w_r = probs['p_w_r']
    p_w = probs['p_w']
    # returns P(R|S, E)
    prob = 1
    # P(R)
    try:
        prob *= p_r[recipient]
    except KeyError:
        return 0
    # P(S|R)
    try:
        prob *= p_s_r[recipient][sender]
    except KeyError:
        return 0
    # P(E|R, S)
    words = tools.get_tokens(email)
    for word in words:
        temp = 0
        try:
            temp += a * p_w_r_s[recipient][sender][word]
        except KeyError:
            pass
        try:
            temp += b * p_w_r[recipient][word]
        except KeyError:
            pass
        try:
            temp += c * p_w[word]
        except KeyError:
            pass       
        prob *= temp
    return prob

from heapq import heappop, heappush

def compute_results(df, df_info, all_recipients, probs, a=1/3, b=1/3, c=1/3):
    res = {}
    pbar = tqdm(range(len(df)))
    for i in pbar:
        sender = df.loc[i, 'sender']
        mids = df.loc[i, 'mids'].split()
        for mid in mids:
            heap = []
            email = list(df_info.loc[df_info['mid'] == int(mid), 'body'])[0]
            # only keep top 10 receivers
            for recipient in all_recipients[:10]:
                prob = predict(recipient, sender, email, probs, a, b, c) 
                heappush(heap, (prob, recipient))
            for recipient in all_recipients[10:]:
                prob = predict(recipient, sender, email, probs, a, b, c) 
                heappush(heap, (prob, recipient))                     
                heappop(heap)
        res[mid] = heap
    return res