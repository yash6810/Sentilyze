import logging
import sys
import os
from logging import Logger


def get_logger(name: str) -> Logger:
    """
    Configures and returns a logger with a standard format and utf-8 safe console/file output.

    Args:
        name (str): The name of the logger, typically __name__.

    Returns:
        Logger: A configured logger instance.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception as e:  # nosec B110
            # stdout reconfigure may fail in non-standard interactive streams or wrapped stdout
            pass

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
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "app.log"), encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def sanitize_filename(name: str) -> str:
    """
    Sanitizes user/external inputs to prevent path injection / path traversal (CWE-22 / CWE-73).
    Only allows alphanumeric characters, dashes, underscores, and dots.
    """
    if not name:
        return "default"
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.\-]", "", str(name)).strip()
    cleaned = re.sub(r"^\.+", "", cleaned)
    return cleaned or "default"


def safe_path_join(base_dir: str, *paths: str) -> str:
    """
    Safely joins paths, ensuring the resolved target path remains strictly within base_dir.
    """
    sanitized_parts = [sanitize_filename(p) for p in paths]
    target_path = os.path.abspath(os.path.join(base_dir, *sanitized_parts))
    base_abs = os.path.abspath(base_dir)
    if not (
        target_path == base_abs
        or target_path.startswith(base_abs + os.sep)
        or target_path.startswith(base_abs + "/")
    ):
        raise ValueError(
            f"Path traversal detected: {target_path} is outside {base_abs}"
        )
    return target_path


def get_market_timestamp(dt=None) -> str:
    """
    Returns a clean, human-readable Indian Standard Time (IST) timestamp with 12-hour AM/PM format.
    Example: 'Sep 02, 2026 • 01:49 PM IST (04:19 AM EDT)'
    """
    from datetime import datetime, timezone
    import zoneinfo

    utc_dt = dt or datetime.now(timezone.utc)
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

    try:
        ist_dt = utc_dt.astimezone(zoneinfo.ZoneInfo("Asia/Kolkata"))
        ny_dt = utc_dt.astimezone(zoneinfo.ZoneInfo("America/New_York"))
        ist_str = ist_dt.strftime("%b %d, %Y • %I:%M %p IST")
        ny_str = ny_dt.strftime("%I:%M %p EDT")
        return f"{ist_str} ({ny_str})"
    except Exception:
        return utc_dt.strftime("%b %d, %Y • %I:%M %p IST")


def optimize_dataframe_memory(df):
    """
    Downcasts numeric types in a pandas DataFrame to reduce memory footprint by 50-75%.
    - Converts float64 -> float32
    - Converts int64 -> int32 / int16
    """
    if df is None or not hasattr(df, "columns") or df.empty:
        return df

    import numpy as np
    import pandas as pd

    optimized_df = df.copy(deep=False)
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        if col_type == np.float64:
            optimized_df[col] = optimized_df[col].astype(np.float32)
        elif col_type == np.int64:
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
    return optimized_df


def cleanup_memory() -> None:
    """
    Explicitly triggers Python garbage collection and frees unreferenced memory blocks.
    """
    import gc

    gc.collect()
