import pandas as pd
import src.bayesian as bayes

path_to_data = 'data/'
training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
training_info = pd.read_csv(
    path_to_data + 'training_info.csv', sep=',', header=0)
test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
test_info = pd.read_csv(
    path_to_data + 'test_info.csv', sep=',', header=0)

p_r = bayes.compute_recipient_prior(training_info)

p_s_r = bayes.compute_sender_likelihood_given_recipient(training, training_info)

p_w, p_w_r, p_w_r_s = bayes.compute_mail_likelihood_given_recipient_and_sender(training, training_info)

probs = {}
probs['p_r'] = p_r 
probs['p_s_r'] = p_s_r
probs['p_w_r_s'] = p_w_r_s
probs['p_w_r'] = p_w_r
probs['p_w'] = p_w

mail_probable = list(training_info[training_info['mid'] == 158713].body)[0]
mail_unprobable = list(training_info[training_info['mid'] == 60].body)[0]

print(bayes.predict('karen.buckley@enron.com', 'jason.wolfe@enron.com', mail_probable, probs))
print(bayes.predict('karen.buckley@enron.com', 'jason.wolfe@enron.com', mail_unprobable, probs))

res = bayes.compute_results(training, training_info)
res = bayes.compute_results(test, test_info)

training_info[training_info['mid'] == ].receivers()

