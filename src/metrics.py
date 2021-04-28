import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import log_loss


def spearmanr(target, pred):
    return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]


def numerai_score(y_true, y_pred):
    rank_pred = rankdata(y_pred)
    return np.corrcoef(y_true, rank_pred)[0, 1]


def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def ar1(x):
    x = x.dropna()
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def autocorr_penalty(x):
    x = x.dropna()
    n = len(x)
    p = ar1(x)
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p**i for i in range(1, n)]))


def smart_sharpe(x):
    x = x.dropna()
    return np.mean(x) / (np.std(x, ddof=1) * autocorr_penalty(x))


def numerai_sharpe(x):
    x = x.dropna()
    return ((np.mean(x) - 0.010415154) / np.std(x)) * np.sqrt(12)
