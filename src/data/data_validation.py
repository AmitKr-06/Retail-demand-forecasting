# pandera schema checks, price checks, value check after saving

import pandas as pd
import numpy as np
import pandera.pandas as pa

from src.constant import ID_COLS

SALES_WIDE_SCHEMA = pa.DataFrameSchema(
    {**{c: pa.Column(object) for c in ID_COLS},
     r"^d_\d+$": pa.Column("int16", pa.Check.ge(0), regex=True)},
    unique=["item_id"], strict=True)

CALENDAR_SCHEMA = pa.DataFrameSchema(
    {"date": pa.Column(pa.DateTime),
     "wm_yr_wk": pa.Column(int),
     "weekday": pa.Column(object),
     "wday": pa.Column(int, pa.Check.isin(list(range(1, 8)))),
     "month": pa.Column(int, pa.Check.isin(list(range(1, 13)))),
     "year": pa.Column(int),
     "d": pa.Column(object),
     "event_name_1": pa.Column(object, nullable=True),
     "event_type_1": pa.Column(object, nullable=True),
     "event_name_2": pa.Column(object, nullable=True),
     "event_type_2": pa.Column(object, nullable=True),
     "snap_CA": pa.Column(int, pa.Check.isin([0, 1])),
     "snap_TX": pa.Column(int, pa.Check.isin([0, 1])),
     "snap_WI": pa.Column(int, pa.Check.isin([0, 1]))},
    unique=["d"], strict=True)

PRICES_SCHEMA = pa.DataFrameSchema(
    {"store_id": pa.Column(object),
     "item_id": pa.Column(object),
     "wm_yr_wk": pa.Column(int),
     "sell_price": pa.Column(float, pa.Check.gt(0), nullable=False)},
    unique=["item_id", "wm_yr_wk"], strict=True)


def long_schema(n_days: int):
    """Checks on the merged long table (columns not listed here are not checked)."""
    return pa.DataFrameSchema(
        {"sales": pa.Column("int16", pa.Check.ge(0)),
         "d_num": pa.Column("int16", pa.Check.in_range(1, n_days)),
         "wday": pa.Column("int8", pa.Check.in_range(1, 7)),
         "month": pa.Column("int8", pa.Check.in_range(1, 12)),
         "snap": pa.Column("int8", pa.Check.isin([0, 1])),
         "sell_price": pa.Column("float32", pa.Check.gt(0), nullable=True)},
        unique=["item_id", "d_num"])


def validate_sales_wide(df):
    return SALES_WIDE_SCHEMA.validate(df)


def validate_calendar(df):
    return CALENDAR_SCHEMA.validate(df)


def validate_prices(df):
    return PRICES_SCHEMA.validate(df)


def check_price_gaps(df: pd.DataFrame) -> int:
    """
    A price may be missing only BEFORE an item's first price (item not on sale yet).
    Returns the number of missing prices after an item's first price (must be 0).
    Sorts internally so the result is correct regardless of input row order.
    """
    df_sorted = df.sort_values(["item_id", "d_num"])
    has_price = df_sorted["sell_price"].notna()
    seen = has_price.astype("int8").groupby(df_sorted["item_id"], observed=True).cummax()
    return int(((seen == 1) & (~has_price)).sum())


def check_price_values(df: pd.DataFrame, prices_s: pd.DataFrame) -> None:
    """
    Every price in the long table must equal the source price of that item and week,
    and a price may exist only where the source table has one (never filled, never overwritten).

    Also checks that each (item_id, wm_yr_wk) pair in df has exactly one distinct price,
    so an intra-week inconsistency cannot hide behind drop_duplicates.
    """
    # 1) one distinct price per (item, week) in df
    nunique = df.groupby(["item_id", "wm_yr_wk"], observed=True)["sell_price"].nunique()
    bad = nunique[nunique > 1]
    assert bad.empty, f"{len(bad)} (item, week) pairs have inconsistent prices"

    # 2) every (item, week) present in df must match the source value, and vice versa
    got = df[["item_id", "wm_yr_wk", "sell_price"]].drop_duplicates(["item_id", "wm_yr_wk"]).copy()
    got["item_id"] = got["item_id"].astype(str)   # df["item_id"] is category, source is object

    src = prices_s[["item_id", "wm_yr_wk", "sell_price"]].copy()
    src["item_id"] = src["item_id"].astype(str)
    src["sell_price"] = src["sell_price"].astype("float32")      # same rounding as the long table

    m = got.merge(src, on=["item_id", "wm_yr_wk"], how="left", suffixes=("", "_src"))
    has, has_src = m["sell_price"].notna(), m["sell_price_src"].notna()
    assert (has == has_src).all(), \
        "a price exists where the source has none, or is missing where it has one"
    assert np.array_equal(m.loc[has, "sell_price"].to_numpy(),
                          m.loc[has, "sell_price_src"].to_numpy()), \
        "prices differ from the source table"


def validate_long(df: pd.DataFrame, n_items: int, n_days: int):
    """One row per item and day (unique key + exact row count), valid ranges, no price gaps."""
    long_schema(n_days).validate(df, lazy=True)
    assert len(df) == n_items * n_days, f"expected {n_items * n_days} rows, got {len(df)}"
    gaps = check_price_gaps(df)
    assert gaps == 0, f"{gaps} missing prices after an item's first price"
    return df


def validate_saved(path, df: pd.DataFrame) -> None:
    """
    Reload the saved file and compare every column, dtype and value exactly.
    parquet round-trips preserve float32 values and category codes, so this must pass.
    """
    chk = pd.read_parquet(path)
    pd.testing.assert_frame_equal(chk, df, check_dtype=True)