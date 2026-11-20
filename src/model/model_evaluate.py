# backtest, MAE/WAPE/MASE, coverage
# metrics, backtest folds and the backtest function: the same yardstick for every model

import warnings
import numpy as np
import pandas as pd

from src.config import PARAMS
from src.constant import PROCESSED_DIR, PROCESSED_LONG_FILE, MASE_SEASON
from src.logger import logging

HORIZON = int(PARAMS["forecast"]["horizon"])


def load_sales_matrix():
    """
    Item x day matrix of sales. Days before an item's first price (item not on sale yet) stay NaN.
    Returns the matrix (DataFrame, rows sorted by item_id) and the category of each row.
    """
    cols = ["item_id", "d_num", "sales", "sell_price", "cat_id"]
    df = pd.read_parquet(PROCESSED_DIR / PROCESSED_LONG_FILE, columns= cols)
    total = int(df["sales"].sum())
    df["sales"] = df["sales"].astype("float32").where(df["sell_price"].notna())
    Y = df.pivot(index="item_id", columns="d_num", values="sales")
    Y = Y.reindex(columns=range(1, int(df["d_num"].max()) + 1))
    assert int(np.nansum(Y.to_numpy())) == total, "units changed while building the matrix"
    cat = (df.drop_duplicates("item_id").set_index("item_id")["cat_id"]
           .reindex(Y.index).astype(str).to_numpy())
    return Y, cat


def final_window(last_day: int, horizon: int = HORIZON):
    """The last 'horizon' days: the final test window, Returns (first test day, cutoff)."""
    test_start = last_day - horizon + 1
    return test_start, test_start - 1


def make_folds(last_day: int, horizon: int = HORIZON, n_folds: int | None = None) -> pd.DataFrame:
    """Consecutive validation windows that end right before the final test window."""
    n_folds = int(PARAMS["forecast"]["n_folds"]) if n_folds is None else int(n_folds)
    test_start, _ = final_window(last_day, horizon)
    rows = []
    for i in range(1, n_folds + 1):
        val_end = test_start - 1 - (n_folds - i) * horizon
        val_start = val_end - horizon + 1
        rows.append({"fold": i, "train_end": val_start - 1, "val_start": val_start, "val_end": val_end})
    return pd.DataFrame(rows)


# metrics (NaN actuals, i.e days before an exists, are ignored)
def _valid(y, p):
    m = ~np.isnan(y) & ~np.isnan(p)
    return y[m], p[m]

def mae(y, p):
    y, p = _valid(y, p)
    return float(np.abs(y - p).mean())

def rmse(y, p):
    y, p = _valid(y, p)
    return float(np.sqrt(((y - p) ** 2).mean()))

def wape(y, p):
    y, p = _valid(y, p)
    return float(np.abs(y - p).sum() / y.sum())

def mase_scale(Y_train, m: int = MASE_SEASON):
    """Per-item mean absolute error of the seasonal naive forecast on the training data."""
    d = np.abs(Y_train[:, m:] - Y_train[:, :-m])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        return np.nanmean(d, axis=1)


def mase(y, p, scale):
    """Mean over items of (item MAE / item scale); returns it and the number of items skipped."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category= RuntimeWarning)
        ratio = np.nanmean(np.abs(y - p), axis = 1) / scale
    ok = np.isfinite(ratio) & (scale > 0)
    return float(ratio[ok].mean()), int((~ok).sum())

def score(Y_val, P, Y_train) -> dict:
    m, skipped = mase(Y_val, P, mase_scale(Y_train))
    return {"MAE": mae(Y_val, P), "RMSE": rmse(Y_val, P), "WAPE": wape(Y_val, P),
            "MASE": m, "mase_items_skipped": skipped}

def backtest(forecaster, name: str, Y_all, folds: pd.DataFrame, h: int = HORIZON, **kw) -> pd.DataFrame:
    """
    Runs any forecaster over the folds. A forecaster takes Y_train (items x days) and returns an
    items x h forecast. It only ever sees the days up to the fold's cutoff.
    """
    last_allowed = Y_all.shape[1] - h               # the final test window is never used
    out = []
    for _, f in folds.iterrows():
        te, vs, ve = int(f["train_end"]), int(f["val_start"]), int(f["val_end"])
        assert vs == te + 1 and ve - vs + 1 == h, "folds windows are not consecutive"
        assert ve <= last_allowed, "a fold reaches into the final test window"
        Y_train, Y_val = Y_all[:, :te], Y_all[:, vs -1: ve]
        P = forecaster(Y_train, h=h, **kw)
        assert P.shape == Y_val.shape, f"forecast shape {P.shape} != {Y_val.shape}"
        out.append({"model": name, "fold": int(f["fold"]), **score(Y_val, P, Y_train)})
    return pd.DataFrame(out)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """Mean of the folds one row per model."""
    return res.groupby("model", sort=False)[["MAE", "RMSE", "WAPE", "MASE"]].mean()