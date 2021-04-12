import numpy as np
from scipy.stats import rankdata


def numerai_score(y_true, y_pred):
    rank_pred = rankdata(y_pred)
    return np.corrcoef(y_true, rank_pred)[0, 1]


def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
