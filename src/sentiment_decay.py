"""
Temporal Sentiment Half-Life Decay Engine.

Functions:
- Applies continuous exponential time-decay weighting to financial news headlines.
- Formula: w_i = exp(-delta_t_i / tau), where tau = half_life_hours / ln(2).
- Ensures breaking catalysts published immediately prior to market open receive up to 4x
  the weighting of older, already-priced-in yesterday headlines.
"""

from typing import Dict, Any, List, Optional
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger

logger = get_logger(__name__)


def compute_exponential_decay_weights(
    timestamps: pd.Series,
    eval_time: Optional[datetime] = None,
    half_life_hours: float = 4.0,
) -> np.ndarray:
    """
    Computes normalized exponential decay weights for an array/Series of timestamps.
    """
    if timestamps.empty:
        return np.array([])

    if eval_time is None:
        eval_time = datetime.now(timezone.utc)

    # Convert timestamps to UTC datetime if not already
    ts_series = pd.to_datetime(timestamps, utc=True)
    eval_ts = pd.to_datetime(eval_time, utc=True)

    delta_hours = (eval_ts - ts_series).dt.total_seconds() / 3600.0
    # Guard against future dates
    delta_hours = delta_hours.clip(lower=0.0)

    # tau = half_life / ln(2)
    tau = half_life_hours / math.log(2.0)
    raw_weights = np.exp(-delta_hours.values / (tau + 1e-8))

    sum_weights = np.sum(raw_weights)
    if sum_weights > 0:
        norm_weights = raw_weights / sum_weights
    else:
        norm_weights = np.ones(len(raw_weights)) / len(raw_weights)

    return norm_weights


def calculate_time_decayed_sentiment(
    news_df: pd.DataFrame,
    score_col: str = "sentiment_score",
    time_col: str = "date",
    half_life_hours: float = 4.0,
    eval_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Calculates the exponential time-decayed aggregate sentiment score across news items.
    """
    if news_df.empty or score_col not in news_df.columns:
        return {
            "decayed_sentiment_score": 0.0,
            "raw_mean_score": 0.0,
            "headline_count": 0,
            "half_life_hours": half_life_hours,
            "temporal_acceleration": 0.0,
        }

    valid_df = news_df.dropna(subset=[score_col]).copy()
    if valid_df.empty:
        return {
            "decayed_sentiment_score": 0.0,
            "raw_mean_score": 0.0,
            "headline_count": 0,
            "half_life_hours": half_life_hours,
            "temporal_acceleration": 0.0,
        }

    raw_mean = float(valid_df[score_col].mean())

    # If timestamp column available, compute decay weights
    if time_col in valid_df.columns:
        try:
            weights = compute_exponential_decay_weights(
                valid_df[time_col],
                eval_time=eval_time,
                half_life_hours=half_life_hours,
            )
            decayed_score = float(np.dot(weights, valid_df[score_col].values))
        except Exception as e:
            logger.debug(f"Sentiment time decay calculation fallback: {e}")
            decayed_score = raw_mean
    else:
        decayed_score = raw_mean

    temporal_accel = round(decayed_score - raw_mean, 4)

    return {
        "decayed_sentiment_score": round(decayed_score, 4),
        "raw_mean_score": round(raw_mean, 4),
        "headline_count": len(valid_df),
        "half_life_hours": half_life_hours,
        "temporal_acceleration": temporal_accel,
    }
