"""
Unit tests for AI Chart Pattern Recognition and Visual Learning Engine.
"""

import pytest
import numpy as np
import pandas as pd
from src.chart_pattern_learning import (
    normalize_waveform,
    detect_classical_chart_patterns,
    match_historical_chart_twins,
    generate_ai_chart_explanation,
)


def test_normalize_waveform():
    series = np.array([10, 20, 30, 40, 50])
    norm = normalize_waveform(series, target_length=10)
    assert len(norm) == 10
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[-1], 1.0)


def test_detect_classical_chart_patterns():
    # Construct synthetic Double Bottom: 100 -> 90 -> 95 -> 90 -> 102
    dates = pd.date_range("2026-01-01", periods=30)
    prices = [100.0] * 5 + [90.0] * 5 + [96.0] * 5 + [90.2] * 5 + [103.0] * 10
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": [2000] * 30,
        },
        index=dates,
    )

    patterns = detect_classical_chart_patterns(df)
    assert isinstance(patterns, list)


def test_match_historical_chart_twins():
    dates = pd.date_range("2026-01-01", periods=30)
    prices = np.linspace(100, 130, 30)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": [2000] * 30,
        },
        index=dates,
    )

    twins = match_historical_chart_twins(df)
    assert "closest_pattern" in twins
    assert "similarity_pct" in twins
    assert twins["similarity_pct"] >= 50.0


def test_generate_ai_chart_explanation():
    dates = pd.date_range("2026-01-01", periods=30)
    prices = np.linspace(100, 130, 30)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": [2000] * 30,
        },
        index=dates,
    )

    smart_zones = {"volume_poc": 115.0, "market_structure": "BULLISH_EXPANSION"}
    twins = match_historical_chart_twins(df)
    patterns = detect_classical_chart_patterns(df)

    expl = generate_ai_chart_explanation("NVDA", df, patterns, twins, smart_zones)
    assert "NVDA" in expl
    assert "AI Visual Chart Story" in expl
