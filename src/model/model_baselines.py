# naive, seasonal naive, moving average
# simple forecasting methods: the yardstick every model must beat

import sys
import warnings
import numpy as np
import pandas as pd

from src.constant import REPORTS_DIR, BASELINE_RESULTS_FILE, MA_WINDOW, MASE_SEASON
from src.exception import CustomException
from src.logger import logging
from src.model.model_evaluate import HORIZON, load_sales_matrix, make_folds, backtest, summarize


def _rep(v, h):
    """One value per item, repeated for every forecast day; items without history get 0."""
    return np.repeat(np.nan_to_num(v, nan=0.0)[:, None], h, axis=1)


def zeros(Y_train, h=HORIZON):
    return np.zeros((Y_train.shape[0], h), dtype="float32")


def naive(Y_train, h=HORIZON):
    return _rep(Y_train[:, -1], h)


def seasonal_naive(Y_train, h=HORIZON, m=MASE_SEASON):
    fc = np.tile(Y_train[:, -m:], (1, int(np.ceil(h / m))))[:, :h]
    return np.nan_to_num(fc, nan=0.0)


def moving_average(Y_train, h=HORIZON, w=MA_WINDOW):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return _rep(np.nanmean(Y_train[:, -w:], axis=1), h)


def moving_median(Y_train, h=HORIZON, w=MA_WINDOW):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return _rep(np.nanmedian(Y_train[:, -w:], axis=1), h)


def empirical_quantile(Y_train, a, h=HORIZON, w=MA_WINDOW):
    """The a-quantile of each item's last w days, repeated for every forecast day."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return _rep(np.nanquantile(Y_train[:, -w:], a, axis=1), h)


METHODS = {"zeros (reference)": zeros, "naive": naive, "seasonal naive (7d)": seasonal_naive,
           "moving average (28d)": moving_average, "moving median (28d)": moving_median}

def run_baselines():
    try:
        logging.info("baseline started")
        Y, _ = load_sales_matrix()
        Y_np = Y.to_numpy()
        folds = make_folds(Y_np.shape[1])
        logging.info("matrix %s | NaN cells %d | total units %d", Y.shape,
                     int(np.isnan(Y_np).sum()), int(np.nansum(Y_np)))
        logging.info("folds:\n%s", folds.to_string(index=False))

        res = pd.concat([backtest(fn, name, Y_np, folds) for name, fn in METHODS.items()], ignore_index=True)

        z = res.loc[res["model"] == "zeros (reference)", "WAPE"].to_numpy()
        assert np.allclose(z, 1.0), "WAPE of the all-zeros forecast must be exactly 1.0 in every fold"

        logging.info("results, mean of the folds:\n%s", summarize(res).round(3).to_string())
        logging.info("WAPE per fold:\n%s", res.pivot(index="model", columns="fold", values="WAPE").round(3).to_string())

        REPORTS_DIR.mkdir(parents=True, exist_ok= True)
        res.to_csv(REPORTS_DIR / BASELINE_RESULTS_FILE, index=False)
        logging.info("saved %s", REPORTS_DIR / BASELINE_RESULTS_FILE)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_baselines()