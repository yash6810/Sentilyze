"""
Unit tests for Post-Earnings Announcement Drift (PEAD) & SUE Earnings Surprise Radar.
Tests SUE calculation, PEAD momentum classification, and mock earnings reports.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.earnings_whisper import analyze_earnings_surprises


def test_earnings_surprises_fallback_on_empty():
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.get_earnings_history.return_value = pd.DataFrame()
        mock_instance.earnings_history = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        report = analyze_earnings_surprises("XYZ_NONEXISTENT")
        assert report["ticker"] == "XYZ_NONEXISTENT"
        assert report["pead_momentum_signal"] == "NEUTRAL_DRIFT"
        assert report["pead_bias"] == "NEUTRAL"
        assert report["catalyst_conviction_boost"] == 0.0


def test_earnings_surprises_bullish_beat():
    mock_history = pd.DataFrame(
        [
            {
                "reportDate": "2026-08-28",
                "epsActual": 3.50,
                "epsEstimate": 3.00,
                "surprisePercent": 16.67,
            },
            {
                "reportDate": "2026-05-25",
                "epsActual": 2.80,
                "epsEstimate": 2.70,
                "surprisePercent": 3.7,
            },
            {
                "reportDate": "2026-02-20",
                "epsActual": 2.40,
                "epsEstimate": 2.30,
                "surprisePercent": 4.3,
            },
        ]
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.get_earnings_history.return_value = mock_history
        mock_ticker.return_value = mock_instance

        report = analyze_earnings_surprises("NVDA")
        assert report["ticker"] == "NVDA"
        assert report["actual_eps"] == 3.50
        assert report["consensus_eps"] == 3.00
        assert report["eps_surprise_pct"] > 10.0
        assert report["sue_score"] > 0.0
        assert report["pead_bias"] == "BULLISH"
        assert report["catalyst_conviction_boost"] > 0.0


def test_earnings_surprises_bearish_miss():
    mock_history = pd.DataFrame(
        [
            {
                "reportDate": "2026-08-28",
                "epsActual": 1.20,
                "epsEstimate": 1.60,
                "surprisePercent": -25.0,
            },
            {
                "reportDate": "2026-05-25",
                "epsActual": 1.50,
                "epsEstimate": 1.50,
                "surprisePercent": 0.0,
            },
            {
                "reportDate": "2026-02-20",
                "epsActual": 1.40,
                "epsEstimate": 1.40,
                "surprisePercent": 0.0,
            },
        ]
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.get_earnings_history.return_value = mock_history
        mock_ticker.return_value = mock_instance

        report = analyze_earnings_surprises("BAD_STOCK")
        assert report["ticker"] == "BAD_STOCK"
        assert report["actual_eps"] == 1.20
        assert report["consensus_eps"] == 1.60
        assert report["eps_surprise_pct"] < -10.0
        assert report["sue_score"] < 0.0
        assert report["pead_bias"] == "BEARISH"
        assert report["catalyst_conviction_boost"] < 0.0
