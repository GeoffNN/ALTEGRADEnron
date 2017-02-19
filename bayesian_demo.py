import numpy as np
import pandas as pd
import pickle as pkl
import src.tools as tools
import src.bayesian as bayes
from pathlib import Path

path_to_data = 'data/'
training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
training_info = pd.read_csv(
    path_to_data + 'training_info.csv', sep=',', header=0)
test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
test_info = pd.read_csv(
    path_to_data + 'test_info.csv', sep=',', header=0)

data_file = Path(path_to_data + 'data.p')

if data_file.is_file():
    print('Loading probabilities...')
    data = pkl.load(open(path_to_data + 'data.p', 'rb'))  
    print('Done')
else:
    print('Computing probabilities...')
    print('Computing recipient prior')
    p_r = bayes.compute_recipient_prior(training_info)
    print('Computing sender likelihood given recipient')
    p_s_r = bayes.compute_sender_likelihood_given_recipient(training, training_info)
    print('Computing mail likelihood given recipient and sender')
    p_w, p_w_r, p_w_r_s, r_s = bayes.compute_mail_likelihood_given_recipient_and_sender(training, training_info)
    data = {}
    data['p_r'] = p_r 
    data['p_s_r'] = p_s_r
    data['p_w_r_s'] = p_w_r_s
    data['p_w_r'] = p_w_r
    data['p_w'] = p_w
    data['r_s'] = r_s
    pkl.dump(data, open(path_to_data + 'data.p', 'wb'))
    print('Done')
    
res = bayes.compute_results(test, test_info, data)
pkl.dump(res, open(path_to_data + 'test_results.p', 'wb'))
