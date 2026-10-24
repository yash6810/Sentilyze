"""
Unit tests for Hawkes Self-Exciting Point-Process & Endogenous Liquidity Cascade Radar.
"""

import pytest
import numpy as np
from src.hawkes_cascade_engine import (
    HawkesCascadeEngine,
    evaluate_hawkes_cascade_risk,
)


def test_hawkes_engine_initialization():
    engine = HawkesCascadeEngine(decay_beta=1.5, supercritical_threshold=1.0)
    assert engine.beta == 1.5
    assert engine.supercritical_thresh == 1.0
    assert engine.elevated_thresh == 0.70


def test_hawkes_calibration_empty_events():
    engine = HawkesCascadeEngine()
    res = engine.calibrate_kernel_from_events(np.array([]), np.array([]))
    assert res["regime"] == "EXOGENOUS_STABLE"
    assert res["branching_ratio_eta"] < 0.50
    assert res["is_cascade_risk"] is False


def test_hawkes_calibration_supercritical_clustering():
    """
    Simulates rapid, high-frequency clustered buy events with tiny inter-arrival deltas.
    Should trigger elevated or supercritical branching ratio.
    """
    engine = HawkesCascadeEngine(decay_beta=1.0)
    # Rapid clustered bursts of trades (dt = 0.05 seconds)
    rapid_buys = np.array([10.0, 10.05, 10.09, 10.12, 10.14, 10.15, 10.16, 10.17])
    scattered_sells = np.array([5.0, 15.0])

    res = engine.calibrate_kernel_from_events(
        rapid_buys, scattered_sells, observation_window=20.0
    )
    assert res["branching_ratio_eta"] >= 0.70
    assert res["regime"] in ["ENDOGENOUS_SELF_EXCITATION", "SUPERCRITICAL_AVALANCHE"]
    assert res["is_cascade_risk"] is True
    assert res["reflexivity_index"] > 0.0  # Buy pressure reflexivity


def test_hawkes_price_series_evaluation():
    engine = HawkesCascadeEngine()
    # Upward trending series with momentum micro-bursts
    prices = np.array(
        [100.0, 100.2, 100.5, 100.4, 100.8, 101.1, 101.0, 101.5, 102.0, 102.3]
    )
    res = engine.evaluate_synthetic_or_price_series(prices, dt_seconds=1.0)
    assert "branching_ratio_eta" in res
    assert "regime" in res
    assert isinstance(res["alpha_matrix"], list)
    assert 0.0 <= res["branching_ratio_eta"] <= 5.0


def test_evaluate_hawkes_cascade_risk_public_api():
    prices = [210.0 + i * 0.5 for i in range(20)]
    res = evaluate_hawkes_cascade_risk("NVDA", custom_prices=prices)
    assert res["ticker"] == "NVDA"
    assert "branching_ratio_eta" in res
    assert "reflexivity_index" in res
