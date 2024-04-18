import logging
import traceback
from datetime import datetime
from typing import Union, Callable, Any


DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRASH_ON = ERROR
severity_to_name = {
    DEBUG: "DEBUG",
    INFO: "INFO",
    WARNING: "WARNING",
    ERROR: "ERROR",
}


def error_catch(func: Callable, *args, **kwargs) -> Any:
    """
    Any raised Exceptions that are not Keyboard/System interrupts are captured here and then sent to the logger as an
    error so all errors are saved to the .log file.

    Args:
        func (Callable): function to run and catch errors on. All other parameters are input into func.

    Returns:
        Any: function output.
    """
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        error(e)
        raise RuntimeError("Should not reach here")
    return result


def set_log_config(minimum_print_severity: int = INFO, log_file_path: str = None) -> None:
    """
    Set the required information before logging.

    Args:
        minimum_print_severity (int): the minimum severity of message to be printed to the terminal.
        log_file_path (str, optional): the file path to the file to place all messages inside of. Default: do not save.
    """
    global _minimum_print_severity
    _minimum_print_severity = minimum_print_severity
    global _log_file
    _log_file = log_file_path
    logging.basicConfig(format="%(message)s", level=logging.ERROR)
    logging.getLogger("coppafish").setLevel(logging.DEBUG)


def debug(msg: Union[str, Exception]) -> None:
    log(msg, DEBUG)


def info(msg: Union[str, Exception]) -> None:
    log(msg, INFO)


def warn(msg: Union[str, Exception]) -> None:
    log(msg, WARNING)


def error(msg: Union[str, Exception]) -> None:
    log(msg, ERROR)


def log(msg: Union[str, Exception], severity: int) -> None:
    """
    Log a message to the log file. The message is printed to the terminal if the message is severe enough.

    Args:
        msg (str or str like or Exception): message to log. Either a str or something that can be converted into a str.
        severity (int): severity of message.
    """
    message = datetime_string()
    message += f":{severity_to_name[severity]}: "
    message += str(msg)
    if _log_file is not None:
        # Append message to log file
        append_to_log_file(message)
    if severity >= _minimum_print_severity:
        logging.getLogger("coppafish").log(severity, message)
    if severity >= CRASH_ON:
        # Crash on high severity
        if isinstance(msg, Exception):
            # Add the traceback to the log file for debugging purposes
            append_to_log_file(traceback.format_exc())
            raise msg
        raise LogError(message)


def datetime_string() -> str:
    """
    Get the current date/time in a readable format for the logs.

    Returns:
        str: current date and time as a string with second precision.
    """
    return datetime.now().strftime("%d/%m/%y %H:%M:%S.%f")


def append_to_log_file(message: str) -> None:
    """
    Appends `message` to the end of the user's log file, if it exists.

    Args:
        message (str): message to append.
    """
    if _log_file is None:
        return
    with open(_log_file, "a") as log_file:
        log_file.write(message + "\n")


class LogError(Exception):
    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)


set_log_config()