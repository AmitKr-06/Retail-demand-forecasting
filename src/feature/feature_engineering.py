# lags (28+), rolling stats (shifted), calendar, price features
# static features + one feature table per horizon block (sales features shifted by 7/14/21/28 days)

import sys
import numpy as np
import pandas as pd

from src.config import PARAMS
from src.constant import (PROCESSED_DIR, PROCESSED_LONG_FILE, FEATURES_DIR, FEATURES_BLOCK_FILE, FEATURES_META_FILE,
                          TARGET, KEEP_COLS, EVENT_COLS, CAT_FEATURES, STATIC_NUM_FEATURES, PRICE_DIFF_DAYS,
                          PRICE_RATIO_WINDOWS, PRICE_RATIO_MIN_PERIODS, PERTURBATION_DAY, PERTURBATION_ITEMS, PERTURBATION_SEED)
from src.exception import CustomException
from src.logger import logging
from src.utils import timer, read_parquet_checked, save_parquet_verified, save_json, assert_columns

_price_names = {f"price_diff_{PRICE_DIFF_DAYS}", "price_ratio_max",
                *[f"price_ratio_{w}" for w in PRICE_RATIO_WINDOWS]}
assert _price_names <= set(STATIC_NUM_FEATURES), "price feature names and STATIC_NUM_FEATURES disagree"
NAN_OK = {f"price_diff_{PRICE_DIFF_DAYS}", *[f"price_ratio_{w}" for w in PRICE_RATIO_WINDOWS]}


def check_block_shifts(shifts, horizon: int) -> None:
    """Block j predicts horizon days 7(j-1)+1 ... 7j, so its sales features must end 7j days earlier."""
    if horizon % 7 != 0:
        raise ValueError(f"horizon must be a multiple of 7, got {horizon}")
    expected = list(range(7, horizon + 1, 7))
    if [int(s) for s in shifts] != expected:
        raise ValueError(f"block_shifts must be {expected} for a {horizon}-day horizon, got {list(shifts)}")


def sales_feature_names(cfg: dict) -> list:
    return ([f"lag_k{k}" for k in range(int(cfg["n_lags"]))]
            + [f"roll_mean_{w}" for w in cfg["roll_mean_windows"]]
            + [f"roll_std_{w}" for w in cfg["roll_std_windows"]]
            + [f"zero_rate_{int(cfg['zero_rate_window'])}"])


def drop_prelaunch(df: pd.DataFrame) -> pd.DataFrame:
    """Rows before an item's first price (item not on sale yet) are not demand: drop them."""
    first_day = (df.loc[df["sell_price"].notna()]
                 .groupby("item_id", observed=True)["d_num"].min().rename("first_day"))
    df = df.merge(first_day, on="item_id", how="left")
    if df["first_day"].isna().any():
        raise ValueError("some items never have a price")
    df = df[df["d_num"] >= df["first_day"]].reset_index(drop=True)
    assert df["sell_price"].notna().all(), "missing prices after the launch day"
    return df


def add_events_and_age(df: pd.DataFrame) -> pd.DataFrame:
    """No event is a real answer ('none'), not a missing value. age = days since launch."""
    for c in EVENT_COLS:
        df[c] = df[c].astype(object).fillna("none").astype("category")
    df["has_event"] = ((df["event_name_1"] != "none") | (df["event_name_2"] != "none")).astype("int8")
    df["age"] = (df["d_num"] - df["first_day"]).astype("int16")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar facts are known in advance, so same-day values are safe."""
    cal = df.drop_duplicates("d_num")[["d_num", "date"]].sort_values("d_num").set_index("d_num")
    cal["dow"] = cal["date"].dt.dayofweek            # 0 = Monday
    cal["dom"] = cal["date"].dt.day
    cal["woy"] = cal["date"].dt.isocalendar().week
    cal["is_weekend"] = cal["dow"] >= 5
    cal["is_christmas"] = (cal["date"].dt.month == 12) & (cal["date"].dt.day == 25)
    for c in ["dow", "dom", "woy", "is_weekend", "is_christmas"]:
        df[c] = df["d_num"].map(cal[c].astype("int8")).astype("int8")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price now against past prices only. Prices are assumed known at forecast time (planned prices)."""
    gp = df.groupby("item_id", observed=True)["sell_price"]
    df[f"price_diff_{PRICE_DIFF_DAYS}"] = (df["sell_price"] / gp.shift(PRICE_DIFF_DAYS) - 1).astype("float32")
    p_prev = gp.shift(1)
    for w in PRICE_RATIO_WINDOWS:
        prev_mean = (p_prev.groupby(df["item_id"], observed=True)
                     .rolling(w, min_periods=PRICE_RATIO_MIN_PERIODS).mean()
                     .reset_index(level=0, drop=True))
        df[f"price_ratio_{w}"] = (df["sell_price"] / prev_mean).astype("float32")
    df["price_ratio_max"] = (df["sell_price"] / gp.cummax()).astype("float32")
    return df


def add_sales_features(d: pd.DataFrame, shift: int, cfg: dict) -> pd.DataFrame:
    """
    Sales features that END 'shift' days before the target day (shift = 7, 14, 21 or 28).
    lag_k{k} = sales (shift + 7k) days earlier; rolling stats run over sales shifted by 'shift'.
    Needs columns item_id, d_num (consecutive per item) and sales, sorted by item_id and d_num.
    """
    g = d.groupby("item_id", observed=True)["sales"]
    for k in range(int(cfg["n_lags"])):
        d[f"lag_k{k}"] = g.shift(shift + 7 * k).astype("float32")
    z = int(cfg["zero_rate_window"])
    d["_s"] = g.shift(shift)
    d["_z"] = (d["_s"] == 0).astype("float32").where(d["_s"].notna())
    gs = d.groupby("item_id", observed=True)["_s"]
    gz = d.groupby("item_id", observed=True)["_z"]

    def roll(grp, w, fn):
        return (getattr(grp.rolling(w, min_periods=w), fn)()
                .reset_index(level=0, drop=True).astype("float32"))

    for w in cfg["roll_mean_windows"]:
        d[f"roll_mean_{w}"] = roll(gs, int(w), "mean")
    for w in cfg["roll_std_windows"]:
        d[f"roll_std_{w}"] = roll(gs, int(w), "std")
    d[f"zero_rate_{z}"] = roll(gz, z, "mean")
    return d.drop(columns=["_s", "_z"])


def check_no_future_dependence(sub: pd.DataFrame, shift: int, cfg: dict,
                               cut: int = PERTURBATION_DAY, claim: int | None = None):
    """
    Replaces all sales from day 'cut' on with 999 and recomputes the sales features.
    Features before day cut + claim (claim defaults to shift) must not change; later ones must.
    Returns (unchanged_before, changed_after).
    """
    claim = shift if claim is None else claim
    cols = sales_feature_names(cfg)
    a = add_sales_features(sub.copy(), shift, cfg)
    pert = sub.copy()
    pert.loc[pert["d_num"] >= cut, "sales"] = 999
    b = add_sales_features(pert, shift, cfg)
    early = a["d_num"] < cut + claim
    unchanged = np.allclose(a.loc[early, cols].to_numpy("float64"),
                            b.loc[early, cols].to_numpy("float64"), equal_nan=True)
    changed = not np.allclose(a.loc[~early, cols].to_numpy("float64"),
                              b.loc[~early, cols].to_numpy("float64"), equal_nan=True)
    return unchanged, changed


def _min_count(n: int, counts: np.ndarray) -> int:
    return int(np.minimum(n, counts).sum())


def validate_static(df: pd.DataFrame, counts: np.ndarray) -> None:
    expected = {f"price_diff_{PRICE_DIFF_DAYS}": _min_count(PRICE_DIFF_DAYS, counts), "price_ratio_max": 0}
    for w in PRICE_RATIO_WINDOWS:
        expected[f"price_ratio_{w}"] = _min_count(PRICE_RATIO_MIN_PERIODS, counts)
    for col, exp in expected.items():
        got = int(df[col].isna().sum())
        assert got == exp, f"{col}: NaN rows {got} != expected {exp}"
    xmas = int(df.loc[df["is_christmas"] == 1, "sales"].sum())
    if xmas:
        logging.warning("units sold on 25 December: %d (the store was expected to be closed)", xmas)


def validate_block(d: pd.DataFrame, shift: int, counts: np.ndarray, cfg: dict) -> None:
    """NaN rows of every sales feature must equal exactly the rows without enough history."""
    z = int(cfg["zero_rate_window"])
    expected = {f"lag_k{k}": _min_count(shift + 7 * k, counts) for k in range(int(cfg["n_lags"]))}
    expected.update({f"roll_mean_{w}": _min_count(shift + int(w) - 1, counts) for w in cfg["roll_mean_windows"]})
    expected.update({f"roll_std_{w}": _min_count(shift + int(w) - 1, counts) for w in cfg["roll_std_windows"]})
    expected[f"zero_rate_{z}"] = _min_count(shift + z - 1, counts)
    for col, exp in expected.items():
        got = int(d[col].isna().sum())
        assert got == exp, f"shift {shift}, {col}: NaN rows {got} != expected {exp}"
    no_nan = [c for c in STATIC_NUM_FEATURES + CAT_FEATURES if c not in NAN_OK]
    assert int(d[no_nan].isna().sum().sum()) == 0, "unexpected NaN in a feature without history"


def run_feature_engineering():
    try:
        logging.info("feature engineering started")
        cfg = PARAMS["feature"]
        horizon = int(PARAMS["forecast"]["horizon"])
        shifts = [int(s) for s in cfg["block_shifts"]]
        check_block_shifts(shifts, horizon)

        long_df = read_parquet_checked(PROCESSED_DIR / PROCESSED_LONG_FILE)
        last_day = int(long_df["d_num"].max())
        n_prelaunch = int(long_df["sell_price"].isna().sum())

        df = drop_prelaunch(long_df)
        assert len(df) == len(long_df) - n_prelaunch, "dropped rows differ from the rows without a price"
        del long_df
        df = df.sort_values(["item_id", "d_num"]).reset_index(drop=True)

        g = df.groupby("item_id", observed=True)["d_num"].agg(["min", "max", "count"])
        assert ((g["max"] - g["min"] + 1) == g["count"]).all(), "an item has gaps in its days"
        assert (g["max"] == last_day).all(), "an item does not end on the last day"
        counts = g["count"].to_numpy()
        logging.info("launched rows: %d (dropped %d before launch) | items %d | last day d_%d",
                     len(df), n_prelaunch, len(g), last_day)

        with timer("static features"):
            df = add_events_and_age(df)
            df = add_calendar_features(df)
            df = add_price_features(df)
        validate_static(df, counts)

        # sample for the future-dependence check
        first = df.groupby("item_id", observed=True)["first_day"].first()
        ids = pd.Series(first[first <= PERTURBATION_DAY // 2].index).sample(
            PERTURBATION_ITEMS, random_state=PERTURBATION_SEED)
        sub = df.loc[df["item_id"].isin(ids), ["d_num", "item_id", "sales"]].reset_index(drop=True)
        sub["item_id"] = sub["item_id"].astype(str).astype("category")

        num_cols = STATIC_NUM_FEATURES + sales_feature_names(cfg)
        out_cols = KEEP_COLS + num_cols + CAT_FEATURES
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        for shift in shifts:
            unchanged, changed = check_no_future_dependence(sub, shift, cfg)
            assert unchanged and changed, f"shift {shift}: future dependence check failed ({unchanged}, {changed})"
            logging.info("shift %d: no feature changed before day %d, changes afterwards: OK",
                         shift, PERTURBATION_DAY + shift)

            with timer(f"sales features, shift {shift}"):
                df = add_sales_features(df, shift, cfg)
            validate_block(df, shift, counts, cfg)
            assert_columns(df, out_cols, f"feature table s{shift}")
            logging.info("shift %d: lag_k0 NaN rows %d", shift, int(df["lag_k0"].isna().sum()))

            save_parquet_verified(df[out_cols], FEATURES_DIR / FEATURES_BLOCK_FILE.format(shift=shift))

        meta = {"NUM": num_cols, "CAT": CAT_FEATURES, "TARGET": TARGET, "H": horizon,
                "LAST_DAY": last_day, "blocks": {str(i + 1): s for i, s in enumerate(shifts)}}
        save_json(meta, FEATURES_DIR / FEATURES_META_FILE)
        logging.info("feature engineering complete: %d numeric + %d categorical features, %d blocks",
                     len(num_cols), len(CAT_FEATURES), len(shifts))
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_feature_engineering()