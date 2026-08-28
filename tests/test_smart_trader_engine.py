"""
Unit tests for Institutional Smart Money Market Structure & Trailing Stop Engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.smart_trader_engine import (
    find_swing_pivots,
    calculate_smart_money_zones,
    calculate_structural_trailing_stop,
    evaluate_multi_timeframe_confluence,
)


def test_find_swing_pivots():
    dates = pd.date_range("2026-01-01", periods=30)
    # Create a clear peak at index 10 and a valley at index 20
    prices = np.sin(np.linspace(0, 3 * np.pi, 30)) * 10 + 100
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": np.random.randint(1000, 5000, size=30),
        },
        index=dates,
    )

    swing_highs, swing_lows = find_swing_pivots(df, left_bars=2, right_bars=2)
    assert len(swing_highs) >= 1
    assert len(swing_lows) >= 1


def test_smart_money_zones_and_poc():
    dates = pd.date_range("2026-01-01", periods=40)
    prices = np.linspace(100, 150, 40)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 2.0,
            "Low": prices - 2.0,
            "Close": prices,
            "Volume": np.random.randint(1000, 5000, size=40),
        },
        index=dates,
    )

    zones = calculate_smart_money_zones(df)
    assert "volume_poc" in zones
    assert zones["volume_poc"] > 0
    assert "market_structure" in zones


def test_structural_trailing_stop():
    dates = pd.date_range("2026-01-01", periods=25)
    prices = np.linspace(100, 120, 25)  # +20% run
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": np.random.randint(1000, 5000, size=25),
        },
        index=dates,
    )

    entry_p = 100.0
    initial_sl = 95.0
    curr_p = 115.0  # +15% profit

    new_sl, action = calculate_structural_trailing_stop(curr_p, entry_p, df, initial_sl)
    # Stop must ratchet up above initial stop
    assert new_sl > initial_sl
    # When gain is +15%, SL should be at least Breakeven (100.10)
    assert new_sl >= entry_p


def test_multi_timeframe_confluence():
    dates = pd.date_range("2025-01-01", periods=100)
    prices = np.linspace(100, 200, 100)  # Strong bull trend
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 2.0,
            "Low": prices - 2.0,
            "Close": prices,
            "Volume": np.random.randint(10000, 50000, size=100),
        },
        index=dates,
    )

    confluence = evaluate_multi_timeframe_confluence("TEST", df)
    assert "confluence_score_pct" in confluence
    assert confluence["confluence_score_pct"] >= 65.0
    assert "BULLISH" in confluence["daily_trend"]
