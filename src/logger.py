import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

from src.constant import LOG_DIR, MAX_LOG_SIZE, BACKUP_COUNT

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, LOG_FILE)


def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if getattr(logger, "_df_configured", False):    # already set up in this process
        return

    formatter = logging.Formatter("[ %(asctime)s] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE,
                                       backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger._df_configured = True


configure_logger()