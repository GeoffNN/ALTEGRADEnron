import numpy as np
try:
    from tqdm import tqdm_notebook
except ImportError:
    tqdm_notebook = lambda x: x


def get_random_predictions(test_info, all_recipients, k=10):
    """
    returns random predictions per sender
    """
    predictions_per_sender = {}

    test_mids = list(test_info['mid'])

    test_mids_pbar = tqdm_notebook(test_mids)
    for test_mid in test_mids_pbar:
        random_recipients = np.random.choice(all_recipients, k, replace=False)
        predictions_per_sender[int(test_mid)] = list(random_recipients)
    return predictions_per_sender
