"""
Unit tests for 15-Minute Opening Volatility Shield & Low-of-Day Demand Engine.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from src.opening_range_engine import (
    is_opening_15min_whipsaw_period,
    calculate_15min_opening_range,
    find_low_of_day_pullback_entry,
)


def test_calculate_15min_opening_range():
    dates = pd.date_range("2026-08-28", periods=10)
    df = pd.DataFrame(
        {
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [97.0] * 10,
            "Volume": [1000] * 10,
        },
        index=dates,
    )

    or_res = calculate_15min_opening_range("NVDA", df)
    assert or_res["has_opening_range"] is True
    assert or_res["or_high"] >= 100.0
    assert or_res["or_low"] <= 100.0
    assert or_res["or_mid"] == (or_res["or_high"] + or_res["or_low"]) / 2.0


def test_find_low_of_day_pullback_entry():
    dates = pd.date_range("2026-08-28", periods=10)
    # Price is at 96.0 (near low of 95-105 range)
    df = pd.DataFrame(
        {
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [96.0] * 10,
            "Volume": [1500] * 10,
        },
        index=dates,
    )

    entry = find_low_of_day_pullback_entry(
        ticker="TSM",
        df_history=df,
        volume_ratio=1.35,  # Institutional volume confirmed
    )
    assert "should_buy" in entry
    assert "discount_tier" in entry
