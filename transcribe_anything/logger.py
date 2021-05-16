"""
    Super simple logger implementation.
"""

INFO = 0
DEBUG = 1
ERROR = 2
LEVEL = ERROR


def set_logging_level(val: int) -> None:
    """Sets the logging level"""
    global LEVEL  # pylint: disable=W0603
    if val > ERROR:
        return
    if val < INFO:
        return
    LEVEL = val


def log_debug(msg: str) -> None:
    """log debug messages."""
    if LEVEL <= DEBUG:
        print(msg)


def log_error(msg: str) -> None:
    """log error messages."""
    if LEVEL <= ERROR:
        print(msg)


def log_info(msg: str) -> None:
    """log info messages."""
    if LEVEL <= INFO:
        print(msg)
