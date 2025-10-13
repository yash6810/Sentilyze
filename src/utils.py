import logging
import sys
from logging import Logger


def get_logger(name: str) -> Logger:
    """
    Configures and returns a logger with a standard format.

    Args:
        name (str): The name of the logger, typically __name__.

    Returns:
        Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        # Log to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # TODO: Add file handler to log to a file

    return logger
