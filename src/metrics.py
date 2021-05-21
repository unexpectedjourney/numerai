import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import skew, kurtosis

from src.utils import create_era_correlation


def spearmanr(target, pred, *args, **kwargs):
    return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]


def numerai_score(y_true, y_pred, *args, **kwargs):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.dropna()

    y_true = df.y_true.tolist()
    y_pred = df.y_pred.tolist()

    rank_pred = rankdata(y_pred)
    return np.corrcoef(y_true, rank_pred)[0, 1]


def correlation_score(y_true, y_pred, *args, **kwargs):
    return np.corrcoef(y_true, y_pred)[0, 1]


def ar1(x):
    x = x.dropna()
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def autocorr_penalty(x):
    x = x.dropna()
    n = len(x)
    p = ar1(x)
    return np.sqrt(
        1 + 2 * np.sum([((n - i) / n) * p ** i for i in range(1, n)]))


def smart_sharpe(x):
    x = x.dropna()
    return np.mean(x) / (np.std(x, ddof=1) * autocorr_penalty(x))


def numerai_sharpe(x):
    x = x.dropna()
    return ((np.mean(x) - 0.010415154) / np.std(x)) * np.sqrt(12)


def adj_sharpe(x):
    x = x.dropna()
    return numerai_sharpe(x) * (1 + ((skew(x) / 6) * numerai_sharpe(x)) - (
            (kurtosis(x) - 3) / 24) * (numerai_sharpe(x) ** 2))


def neutralized_numerai_score(y_true, y_pred, exposures, proportion=1.0, *args, **kwargs):
    # feature neutralization
    exposures = np.hstack((exposures, np.array(
        [np.mean(y_pred)] * len(exposures)).reshape(-1, 1)))
    y_pred -= proportion * (exposures @ (np.linalg.pinv(exposures) @ y_pred))
    neutralized_preds = y_pred / y_pred.std()

    # numerai_score
    return numerai_score(y_true, neutralized_preds)


def numerai_score_and_sharpe(y_true, y_pred, exposures, eras=None, *args, **kwargs):
    if eras is None:
        eras = []

    corr_metric = neutralized_numerai_score(y_true, y_pred, exposures)
    era_scores = create_era_correlation(
        y_true,
        y_pred,
        eras,
        spearmanr
    )
    sharpe_metric = smart_sharpe(era_scores)
    return 1.4 * corr_metric + 0.03 * sharpe_metric
