# wide -> long, merge calendar and prices, save parquet

import sys
import pandas as pd

from src.config import PARAMS
from src.constant import (INTERIM_DIR, PROCESSED_DIR, INTERIM_SALES_FILE, INTERIM_CALENDAR_FILE,
                          INTERIM_PRICES_FILE, PROCESSED_LONG_FILE, ID_COLS, DAY_PREFIX, EVENT_COLS)
from src.data.data_validation import (validate_sales_wide, validate_calendar, validate_prices,
                                      validate_long, check_price_values, validate_saved)
from src.exception import CustomException
from src.logger import logging


def _day_number(s: pd.Series) -> pd.Series:
    """ 'd_17' -> 17 """
    return pd.to_numeric(s.astype(str).str.replace(DAY_PREFIX, "", regex=False),
                         errors="raise").astype("int16")

def to_long(sales_s: pd.DataFrame, day_cols: list) -> pd.DataFrame:
    """Wide (one column per day) to long (one row per item and day)."""
    sales_s = sales_s.copy()
    for c in ID_COLS:
        sales_s[c] = sales_s[c].astype("category")
    df = sales_s.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    df["d_num"] = _day_number(df["d"])
    return df.drop(columns="d")


def merge_calendar(df: pd.DataFrame, calendar: pd.DataFrame, state: str) -> pd.DataFrame:
    """Adds date, weekday, month, year, events and the SNAP flag of the store's state."""
    cal = calendar.copy()
    cal["d_num"] = _day_number(cal["d"])
    keep = ["d_num", "date", "wm_yr_wk", "wday", "month", "year", *EVENT_COLS, f"snap_{state}"]
    cal = cal[keep].rename(columns={f"snap_{state}": "snap"})

    n = len(df)
    df = df.merge(cal, on="d_num", how="left")
    assert len(df) == n, "calendar merge changed the row count"
    assert df["date"].notna().all(), "days without a date after the calendar merge"
    return df


def merge_prices(df: pd.DataFrame, prices_s: pd.DataFrame) -> pd.DataFrame:
    """Adds the weekly price. Missing = item not on sale yet; it is keep as NaN, never filled."""
    p = prices_s[["item_id", "wm_yr_wk", "sell_price"]].copy()
    # cast to the same categories as df so the merge stays memory-light and keeps category dtype
    p["item_id"] = pd.Categorical(p["item_id"], categories=df["item_id"].cat.categories)

    n = len(df)
    df = df.merge(p, on=["item_id", "wm_yr_wk"], how="left")
    assert len(df) == n, "price merge chagned the row count"
    return df


def shrink_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Smaller dtypes, then sort: each items's days in order (lags and rolling windows need it)."""
    df["wm_yr_wk"] = df["wm_yr_wk"].astype("int16")
    df["wday"] = df["wday"].astype("int8")
    df["month"] = df["month"].astype("int8")
    df["year"] = df["year"].astype("int16")
    df["snap"] = df["snap"].astype("int8")
    df["sell_price"] = df["sell_price"].astype("float32")

    # guard: a bad cast or a fillna upstream would collapse the NaN count - catch it here
    n_missing = int(df["sell_price"].isna().sum())
    assert n_missing > 1_000_000, (f"sell_price has only {n_missing} NaNs - likely a fill/cast happened upstram")
    assert (df["sell_price"].dropna() > 0).all(), "zero or negative price found"

    for c in EVENT_COLS:
        df[c] = df[c].astype("category")
    return df.sort_values(["item_id", "d_num"]).reset_index(drop=True)


def run_preprocessing():
    try:
        logging.info("data preprocessing started")
        sales_s = validate_sales_wide(pd.read_parquet(INTERIM_DIR / INTERIM_SALES_FILE))
        calendar = validate_calendar(pd.read_parquet(INTERIM_DIR / INTERIM_CALENDAR_FILE))
        prices_s = validate_prices(pd.read_parquet(INTERIM_DIR / INTERIM_PRICES_FILE))
        logging.info("interim files passed validation")

        states = sales_s["state_id"].astype(str).unique()
        assert len(states) == 1, f"expected one state, found {states}"
        state = states[0]

        day_cols = [c for c in sales_s.columns if c.startswith(DAY_PREFIX)]
        n_items, n_days = len(sales_s), len(day_cols)
        raw_units = int(sales_s[day_cols].sum().sum())

        df = to_long(sales_s, day_cols)
        df = merge_calendar(df, calendar, state)
        df = merge_prices(df, prices_s)
        df = shrink_and_sort(df)

        validate_long(df, n_items, n_days)
        check_price_values(df, prices_s)
        assert int(df["sales"].sum()) == raw_units, "units changed during preprocessing"

        missing = int(df["sell_price"].isna().sum())
        logging.info("long table: %s | items %d | days %d | state %s", df.shape, n_items, n_days, state)
        logging.info("date range: %s to %s", df["date"].min().date(), df["date"].max().date())
        logging.info("missing prices (before launch): %d (%.1f%%)", missing, 100 * missing / len(df))
        logging.info("total units: %d | memory %.0f MB", int(df["sales"].sum()),
                     df.memory_usage(deep=True).sum() / 1e6)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / PROCESSED_LONG_FILE
        df.to_parquet(out, index=False)
        validate_saved(out, df)
        logging.info("saved %s (%.1f MB), value check passed", out, out.stat().st_size / 1e6)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_preprocessing()