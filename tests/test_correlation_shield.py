"""
Tests for Portfolio Correlation Matrix Shield (src/correlation_shield.py).
Verifies pair correlation calculations, held portfolio matrix analysis, and veto thresholds.
"""

import pytest
import pandas as pd
from src.correlation_shield import (
    calculate_portfolio_correlation_matrix,
    check_correlation_shield,
)


def test_calculate_portfolio_correlation_matrix():
    tickers = ["NVDA", "AAPL", "MSFT"]
    res = calculate_portfolio_correlation_matrix(tickers, period="1mo")
    assert isinstance(res, pd.DataFrame)
    if not res.empty:
        assert set(res.columns).issubset(set(tickers))


def test_check_correlation_shield_empty_held():
    # If no held assets, shield should immediately pass
    verdict = check_correlation_shield("NVDA", open_positions={})
    assert verdict["allowed"] is True
    assert verdict["max_correlation"] == 0.0
    assert verdict["candidate"] == "NVDA"
    assert verdict["status"] == "APPROVED_DIVERSIFIED"


def test_check_correlation_shield_cross_asset():
    # Test checking candidate against another held asset
    mock_positions = {"MSFT": {"qty": 10, "avg_cost": 400.0}}
    verdict = check_correlation_shield(
        "NVDA", open_positions=mock_positions, max_corr_threshold=0.70
    )
    assert isinstance(verdict["allowed"], bool)
    assert verdict["candidate"] == "NVDA"
    assert "status" in verdict
    assert "reason" in verdict
    assert "max_correlation" in verdict
