import logging
from src.utils import get_logger

def test_get_logger():
    """Tests that the get_logger function returns a configured logger."""
    logger_name = "test_logger"
    logger = get_logger(logger_name)

    # Check if the logger has the correct name
    assert logger.name == logger_name

    # Check if the logger has a handler configured
    assert len(logger.handlers) > 0

    # Check if the logger's level is set to INFO
    assert logger.level == logging.INFO