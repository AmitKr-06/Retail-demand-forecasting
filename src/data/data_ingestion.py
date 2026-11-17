# read raw CSVs, filter to one store
import sys
import pandas as pd

from src.config import PARAMS
from src.constant import (RAW_DIR, INTERIM_DIR, RAW_SALES_FILE, RAW_CALENDAR_FILE, RAW_PRICES_FILE,
                          INTERIM_SALES_FILE, INTERIM_CALENDAR_FILE, INTERIM_PRICES_FILE,
                          ID_COLS, DAY_PREFIX)
from src.exception import CustomException
from src.logger import logging


def load_raw():
    """Reads the three raw kaggle files."""
    sales = pd.read_csv(RAW_DIR / RAW_SALES_FILE)
    calendar = pd.read_csv(RAW_DIR / RAW_CALENDAR_FILE, parse_dates=["date"])
    prices = pd.read_csv(RAW_DIR / RAW_PRICES_FILE)
    logging.info("raw file loaded: sales %s | calendar %s | prices %s",
                 sales.shape, calendar.shape, prices.shape)
    return sales, calendar, prices

def filter_store(sales: pd.DataFrame, prices: pd.DataFrame, store_id: str):
    """Keeps one store: all 30,490 series would give about 59 million rows once reshaped."""
    if store_id not in set(sales["store_id"].unique()):
        raise ValueError(f"store {store_id!r} not found in the sales file")
    sales_s = sales[sales["store_id"] == store_id].reset_index(drop=True)
    prices_s = prices[prices["store_id"] == store_id].reset_index(drop=True)
    return sales_s, prices_s

def shrink_sales(sales_s: pd.DataFrame):
    """Day columns to int16 (sales are small whole numbers); returns the table and the day columns."""
    day_cols = [c for c in sales_s.columns if c.startswith(DAY_PREFIX)]
    values = sales_s[day_cols].to_numpy()
    if values.min() < 0 or values.max() > 32767:
        raise ValueError("sales outside the int16 range, not converting")
    sales_s[day_cols] = sales_s[day_cols].astype("int16")
    return sales_s, day_cols

def run_ingestion():
    try:
        store_id = PARAMS["data"]["store_id"]
        logging.info("data ingestion started for store %s", store_id)

        sales, calendar, prices = load_raw()
        sales_s, prices_s = filter_store(sales, prices, store_id)
        del sales
        sales_s, day_cols = shrink_sales(sales_s)

        assert all(c in sales_s.columns for c in ID_COLS), "id columns missing"
        assert sales_s["item_id"].is_unique, "an item appears twice in the store"

        logging.info("sales (%s): %s | day columns: %d (%s to %s)", store_id, sales_s.shape,
                     len(day_cols), day_cols[0], day_cols[-1])
        logging.info("prices (%s): %s | calendar: %s", store_id, prices_s.shape, calendar.shape)
        logging.info("total units sold in the store: %d", int(sales_s[day_cols].sum().sum()))


        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        sales_s.to_parquet(INTERIM_DIR / INTERIM_SALES_FILE, index=False)
        prices_s.to_parquet(INTERIM_DIR / INTERIM_PRICES_FILE, index=False)
        calendar.to_parquet(INTERIM_DIR / INTERIM_CALENDAR_FILE, index=False)
        logging.info("saved 3 files in %s", INTERIM_DIR)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_ingestion()