"""Shared helpers: timing, safe file I/O, checks used across notebooks and src modules."""

import time, json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from src.logger import logging


@contextmanager
def timer(label: str):
    """Logs how long block takes.
    
    Usage: with timer("feature build")
                build_features(df)
    """
    start = time.perf_counter()
    logging.info("%s - started", label)
    try:
        yield
    finally:
        logging.info("%s - finished in %.2f s", label, time.perf_counter() - start)


def read_parquet_checked(path, expected_shape: tuple | None=None) -> pd.DataFrame:
    """Loads a parquet file and optionally asserts it's shape."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    df = pd.read_parquet(path)
    if expected_shape is not None:
        assert df.shape == expected_shape, f"{path.name}: expected shape {expected_shape}, got {df.shape}"
    logging.info("loaded %s (%s)", path.name, df.shape)
    return df


def save_parquet_verified(df: pd.DataFrame, path) -> None:
    """
    Saves to parquet, reloads and compare EXACTLY (values, dtypes, categories).
    Parquet keeps flaot32 values unchanged. df must have the default index (it is not saved).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok= True)
    df.to_parquet(path, index=False)

    chk = pd.read_parquet(path)
    pd.testing.assert_frame_equal(chk, df, check_exact=True)
    logging.info("saved and verified: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    del chk


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding = "utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def assert_columns(df: pd.DataFrame, required: list[str], label: str= "") -> None:
    """Fails early if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label or 'dataframe'} missing columns: {missing}")


def assert_no_nan(df: pd.DataFrame, cols: list[str], label: str = "") -> None:
    """Fails early if any of the given columns contain NaN."""
    for c in cols:
        n = int(df[c].isna().sum())
        if n:
            raise ValueError(f"{label or 'dataframe'}: column {c!r} has {n} NaNs")
