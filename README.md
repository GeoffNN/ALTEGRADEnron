# ALTEGRADExon
Code for the MVA ALTEGRAD Course final project

# Project organization


## File usage 

### General information on ipython notebooks

- Each notebook that has the '_val' suffix produces a model and a prediction on the validation set which is composed of roughly the last 2000 emails of the training set

- Each notebook that has the '_model' suffix produces a model and a prediction on the test set, as available in the kaggle contest settings.


- model_fusion.ipynb is used to fusion several models for the test set

- model_fusion_val.ipynb is used to fusion various models for the validation set

The ipython notebooks are constructed to mostly call functions from the relevant src/file to improve readability

### Specific models

- notebooks with the 'recency' prefix use a time decaying scoring for each sender

- notebooks with the 'tree' prefix produce random forests on LDA topic and day-of-week features

- notebooks with the 'twidf' of 'tfidf' prefix use the corresponding representation.

- notebooks that contain 'knn_sender' in their name produce nearest neighbor-based sender-level models

- notebook that contain only 'knn' in their name produce nearest neighbor that search the entire email space (not limited to a given sender).


For instance 'twidf_knn_sender_val.ipynb' produces a model based on twidf that looks for nearest neighbors in the sender space and that is tested on the validation set. 