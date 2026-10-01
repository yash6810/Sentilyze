"""
Unit tests for Triad Cointegration VECM & Stat-Arb Engine (Sprint 2, Module 2.12)
"""

import pytest
import numpy as np
from src.statarb_engine import (
    johansen_cointegration_test,
    fit_ornstein_uhlenbeck,
    analyze_triad_cointegration,
)


def test_johansen_cointegration_synthetic():
    np.random.seed(42)
    n = 200
    # Create cointegrated triad with shared random walk W
    w = np.cumsum(np.random.normal(0, 1, n))
    y1 = w + np.random.normal(0, 0.2, n)
    y2 = 2.0 * w + np.random.normal(0, 0.2, n)
    y3 = 0.5 * w + np.random.normal(0, 0.2, n)

    Y = np.column_stack([y1, y2, y3])
    res = johansen_cointegration_test(Y, p=1)

    assert "trace_statistics" in res
    assert "primary_cointegrating_vector" in res
    assert len(res["primary_cointegrating_vector"]) == 3
    # First component normalized to 1.0
    assert np.isclose(res["primary_cointegrating_vector"][0], 1.0)
    assert res["is_cointegrated"] is True


def test_fit_ornstein_uhlenbeck():
    # Mean-reverting synthetic series: S_t = 0.8 * S_{t-1} + noise
    np.random.seed(42)
    s = [0.0]
    for _ in range(100):
        s.append(0.8 * s[-1] + np.random.normal(0, 0.5))

    ou = fit_ornstein_uhlenbeck(np.array(s))
    assert ou["reversion_speed_theta"] > 0.0
    assert 1.0 <= ou["half_life_days"] <= 10.0
    assert "volatility_sigma" in ou


def test_analyze_triad_cointegration_smoke():
    res = analyze_triad_cointegration(["NVDA", "TSM", "ASML"])
    assert res["status"] == "SUCCESS"
    assert "primary_beta" in res
    assert "spread_z_score" in res
    assert "stat_arb_signal" in res
