from pathlib import Path

# poject root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"

INTERIM_DIR = DATA_DIR / "interim"


# logging
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3                # number of backup log files to keep

# settings file
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

# raw files
RAW_SALES_FILE = "sales_train_evaluation.csv"
RAW_CALENDAR_FILE = "calendar.csv"
RAW_PRICES_FILE = "sell_prices.csv"

# interim files (one store)
INTERIM_SALES_FILE = "sales_store.parquet"
INTERIM_CALENDAR_FILE = "calendar.parquet"
INTERIM_PRICES_FILE = "prices_store.parquet"

# processed files
PROCESSED_LONG_FILE = "sales_long.parquet"      # one row per item and day, calendar prices merged


# raw data layout
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
DAY_PREFIX = "d_"
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

# feature tables
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_BLOCK_FILE = "features_s{shift}.parquet"   # one table per horizon block
FEATURES_META_FILE = "features_blocks_meta.json"

TARGET = "sales"
KEEP_COLS = ["d_num", "date", TARGET]          # kept next to the features in every table
CAT_FEATURES = ["item_id", "dept_id", "cat_id", *EVENT_COLS]

PRICE_DIFF_DAYS = 7
PRICE_RATIO_WINDOWS = (28, 182)      # price vs the mean of the previous N days
PRICE_RATIO_MIN_PERIODS = 28

STATIC_NUM_FEATURES = ["sell_price", "price_diff_7", "price_ratio_28", "price_ratio_max", "price_ratio_182",
                       "age", "dow", "dom", "woy", "month", "is_weekend", "is_christmas", "has_event", "snap"]

PERTURBATION_DAY = 1000         # Feature-dependenc check: sales from this day on are replaced
PERTURBATION_ITEMS = 10
PERTURBATION_SEED = 5