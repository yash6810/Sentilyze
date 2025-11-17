import logging
import sys
import os
from logging import Logger
import pandas as pd
import numpy as np


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

        # Log to file
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_sequences(df: pd.DataFrame, sequence_length: int):
    """
    Transforms a DataFrame into sequences for an LSTM model.
    """
    sequences = []
    labels = []
    # Group by stock ticker to create sequences per stock
    for ticker, group in df.groupby('Ticker'):
        feature_columns = [col for col in df.columns if col not in ['Ticker', 'target']]
        print(f"Feature columns for ticker {ticker}: {feature_columns}")
        features = group[feature_columns].values
        target = group['target'].values

        for i in range(len(features) - sequence_length):
            sequences.append(features[i:i+sequence_length])
            labels.append(target[i+sequence_length])

    return np.array(sequences), np.array(labels)
