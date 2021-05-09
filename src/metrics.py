import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import log_loss

def numerai_score(y_true, y_pred):
    rank_pred = rankdata(y_pred)
    return np.corrcoef(y_true, rank_pred)[0, 1]


def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def cross_entropy_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)


def neutralized_numerai_score(y_true, y_pred, x_val, proportion=1.0):
    # feature neutralization
    exposures = np.hstack((x_val, np.array([np.mean(y_pred)] * len(x_val)).reshape(-1, 1)))
    y_pred -= proportion * (exposures @ (np.linalg.pinv(exposures) @ y_pred))
    neutralized_preds = y_pred / y_pred.std()

    # numerai_score
    return np.corrcoef(y_true, rankdata(neutralized_preds))[0, 1]
