# custom exception (like your Olist one)
import logging
import sys

from src.constant import PROJECT_ROOT


def error_message_detail(error, error_detail: sys) -> str:
    """
    Extracts the file name, line number and message of the  error.
    
    :param error: The exception (or text) that occurred.
    :param error_detail: The sys module, usedd to read the traceback.
    :return: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:          # no active exception: nothing to point  at
        error_message = f"Error: {str(error)}"
    else:
        # deepest frame that is in our own code (not pandas / lightgbm internals)
        own = exc_tb
        tb = exc_tb
        while tb is not None:
            name = tb.tb_frame.f_code.co_filename
            if str(PROJECT_ROOT) in name and "site-package" not in name:
                own = tb
            tb = tb.tb_next
        file_name = own.tb_frame.f_code.co_filename
        line_number = own.tb_lineno
        error_message = (f"Error occurred in python script: [{file_name}]"
                         f"at line number [{line_number}]: {str(error)}")

    logging.error(error_message)
    return error_message

class CustomException(Exception):
    """
    Custom exception for the demand forecasting.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = str(error_message)

        # an already-wrapped error keeps its original location and is not logged twice
        if isinstance(error_message, CustomException):
            self.error_message = error_message.error_message
        else:
            self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return getattr(self, "error_message", super().__str__())