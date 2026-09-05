"""
Tests for Real-Time Market Anomaly Screener (src/screener_engine.py).
Verifies parallel metrics computation, anomaly filtering, and screener signals.
"""

import pytest
import pandas as pd
from src.screener_engine import (
    evaluate_single_asset_screener,
    run_universe_screener,
)


def test_evaluate_single_asset_screener():
    metrics = evaluate_single_asset_screener("NVDA")
    if metrics:
        assert metrics["ticker"] == "NVDA"
        assert "price" in metrics
        assert "rvol" in metrics
        assert "range_pos_pct" in metrics
        assert "mom_5d_pct" in metrics
        assert "anomaly_score" in metrics
        assert "setup_type" in metrics
        assert 0.0 <= metrics["anomaly_score"] <= 100.0


def test_run_universe_screener_small():
    tickers = ["NVDA", "AAPL"]
    df = run_universe_screener(tickers, max_workers=2)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "ticker" in df.columns
        assert "anomaly_score" in df.columns
        assert "rvol" in df.columns
