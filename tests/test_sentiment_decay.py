"""
Tests for Temporal Sentiment Half-Life Decay Engine (src/sentiment_decay.py).
Verifies exponential weighting, freshness factor, and half-life decay.
"""

from datetime import datetime, timezone, timedelta
import pytest
import pandas as pd
import numpy as np
from src.sentiment_decay import (
    compute_exponential_decay_weights,
    calculate_time_decayed_sentiment,
)


def test_compute_exponential_decay_weights():
    now = datetime.now(timezone.utc)
    timestamps = pd.Series(
        [
            now,
            now - timedelta(hours=4),
            now - timedelta(hours=8),
        ]
    )

    weights = compute_exponential_decay_weights(
        timestamps, eval_time=now, half_life_hours=4.0
    )
    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    # The freshest item should have the highest weight
    assert weights[0] > weights[1] > weights[2]
    # Ratio between weight 0 and weight 1 should be approx 2.0 (since delta is 1 half-life)
    assert pytest.approx(weights[0] / weights[1], rel=1e-2) == 2.0


def test_calculate_time_decayed_sentiment():
    now = datetime.now(timezone.utc)
    # 2 headlines: 1 recent positive (+0.8, 0h old), 1 old negative (-0.8, 12h old)
    news_df = pd.DataFrame(
        [
            {"title": "Surge in AI chip sales", "sentiment_score": 0.8, "date": now},
            {
                "title": "Supply chain warning",
                "sentiment_score": -0.8,
                "date": now - timedelta(hours=12),
            },
        ]
    )

    res = calculate_time_decayed_sentiment(news_df, half_life_hours=4.0, eval_time=now)
    assert res["headline_count"] == 2
    # The recent positive headline dominates the decayed negative one
    assert res["decayed_sentiment_score"] > 0.0
    assert res["half_life_hours"] == 4.0
    assert "temporal_acceleration" in res


def test_calculate_time_decayed_sentiment_empty():
    empty_df = pd.DataFrame()
    res = calculate_time_decayed_sentiment(empty_df)
    assert res["headline_count"] == 0
    assert res["decayed_sentiment_score"] == 0.0
