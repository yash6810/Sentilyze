"""
Unit tests for Cross-Asset Credit & Macro Spillover Matrix (Option 2).
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.cross_asset_matrix import (
    fetch_cross_asset_history,
    compute_cross_asset_matrix,
)


def test_fetch_cross_asset_synthetic_fallback():
    df = fetch_cross_asset_history(period="6mo")
    assert not df.empty
    assert len(df) >= 30
    for col in ["SPY", "QQQ", "HYG", "LQD"]:
        assert col in df.columns


def test_compute_cross_asset_matrix_risk_on():
    # Construct mock data simulating Risk-On
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    data = {
        "SPY": np.linspace(500, 550, 40),
        "QQQ": np.linspace(440, 490, 40),
        "HYG": np.linspace(75, 80, 40),
        "LQD": np.linspace(110, 110, 40),  # HYG/LQD ratio increasing
        "TNX": np.linspace(4.2, 4.3, 40),
        "USO": np.linspace(70, 72, 40),
        "GLD": np.linspace(220, 225, 40),
        "BTC": np.linspace(60000, 68000, 40),
    }
    mock_df = pd.DataFrame(data, index=dates)

    res = compute_cross_asset_matrix(history_df=mock_df, save_results=False)
    assert res["status"] == "SUCCESS"
    assert res["credit_ratio_20d_zscore"] > 0
    assert res["regime"] == "RISK_ON_EXPANSION"
    assert res["cro_risk_multiplier"] > 1.0
    assert "SPY" in res["correlation_matrix_30d"]


def test_compute_cross_asset_matrix_credit_stress_divergence():
    # Construct mock data where SPY rallies but HYG crashes relative to LQD
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    spy_vals = np.concatenate([np.linspace(500, 510, 20), np.linspace(510, 540, 20)])
    hyg_vals = np.concatenate(
        [np.linspace(80, 80, 20), np.linspace(80, 70, 20)]
    )  # Sharp drop
    lqd_vals = np.ones(40) * 100.0

    mock_df = pd.DataFrame(
        {
            "SPY": spy_vals,
            "QQQ": spy_vals * 0.9,
            "HYG": hyg_vals,
            "LQD": lqd_vals,
            "TNX": np.ones(40) * 4.3,
            "USO": np.ones(40) * 75.0,
            "GLD": np.linspace(200, 230, 40),  # Gold rallying
            "BTC": np.ones(40) * 60000.0,
        },
        index=dates,
    )

    res = compute_cross_asset_matrix(history_df=mock_df, save_results=False)
    assert res["status"] == "SUCCESS"
    assert res["credit_ratio_20d_zscore"] < -1.0
    assert res["regime"] == "CREDIT_STRESS_DEFENSIVE"
    assert res["cro_risk_multiplier"] < 1.0
    assert len(res["divergence_alerts"]) > 0
