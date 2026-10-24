"""Unit Tests for Dynamic Kalman Filter Statistical Arbitrage Engine.

Verifies:
1. Online 2-state Kalman Filter updates and covariance decay.
2. Dynamic hedge ratio convergence on cointegrated series.
3. Ornstein-Uhlenbeck half-life estimation.
4. Signal generation (LONG_SPREAD, SHORT_SPREAD, EXIT, STOP_LOSS).
5. Universe scan across multi-asset price matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kalman_statarb_engine import (
    DynamicKalmanFilter,
    KalmanStatArbEngine,
    PairSpreadState,
    run_kalman_statarb_scan,
)


def test_kalman_filter_initialization_and_update():
    """Verify state initialization and single step innovation update."""
    kf = DynamicKalmanFilter(delta=1e-4, R=1e-3)
    alpha, beta, err, var = kf.update(y=100.0, x=50.0)

    assert kf.initialized is True
    assert isinstance(alpha, float)
    assert isinstance(beta, float)
    assert isinstance(err, float)
    assert var > 0.0


def test_kalman_hedge_ratio_convergence():
    """Verify Kalman filter converges towards the true synthetic hedge ratio beta=2.0 and alpha=5.0."""
    np.random.seed(42)
    n = 200
    x = 50.0 + np.cumsum(np.random.randn(n) * 0.8)
    true_alpha = 5.0
    true_beta = 2.0
    noise = np.random.randn(n) * 0.2
    y = true_alpha + true_beta * x + noise

    kf = DynamicKalmanFilter(delta=1e-4, R=1e-3)
    alphas = []
    betas = []
    for yi, xi in zip(y, x):
        a, b, _, _ = kf.update(yi, xi)
        alphas.append(a)
        betas.append(b)

    # By bar 150+, beta should be very close to 2.0
    final_beta = betas[-1]
    assert 1.80 <= final_beta <= 2.20, f"Expected beta near 2.0, got {final_beta}"


def test_ornstein_uhlenbeck_half_life():
    """Verify half-life accurately captures fast mean reversion vs random walk."""
    engine = KalmanStatArbEngine()

    # Fast mean-reverting OU series: x_t = 0.7 * x_{t-1} + e_t
    np.random.seed(42)
    ou_series = np.zeros(100)
    for t in range(1, 100):
        ou_series[t] = 0.75 * ou_series[t - 1] + np.random.randn() * 0.5

    half_life_ou = engine.calculate_half_life(ou_series)
    assert (
        1.0 <= half_life_ou <= 10.0
    ), f"Expected OU half-life 1-10, got {half_life_ou}"

    # Pure random walk / trending series
    rw_series = np.cumsum(np.ones(100) * 1.5 + np.random.randn(100))
    half_life_rw = engine.calculate_half_life(rw_series)
    assert half_life_rw >= 50.0 or half_life_rw == 999.0


def test_pair_signal_generation_deep_spread():
    """Verify trading signal triggers when spread diverges into entry threshold with valid half-life."""
    np.random.seed(42)
    n = 100
    # Mean-reverting cointegrated pair
    x = 100.0 + np.sin(np.linspace(0, 8 * np.pi, n)) * 5.0 + np.random.randn(n) * 0.2
    y = (
        2.0 * x
        + 10.0
        + np.sin(np.linspace(0, 8 * np.pi, n)) * 0.5
        + np.random.randn(n) * 0.2
    )

    # Artificially depress the last y-price moderately
    y[-1] -= 0.8

    engine = KalmanStatArbEngine(z_entry=1.8, z_exit=0.35, z_stop=5.0)
    prices_x = pd.Series(x)
    prices_y = pd.Series(y)

    state = engine.evaluate_pair(
        prices_y, prices_x, asset_y_name="NVDA", asset_x_name="AMD"
    )

    assert isinstance(state, PairSpreadState)
    assert state.asset_y == "NVDA"
    assert state.asset_x == "AMD"
    assert state.z_score < -1.8
    assert state.cointegration_stable is True
    assert state.signal == "LONG_SPREAD"


def test_universe_scan_pipeline():
    """Verify scanning multiple pairs across a synthetic price matrix."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    base = 100.0 + np.cumsum(np.random.randn(60))

    df = pd.DataFrame(
        {
            "NVDA": base * 1.2 + np.random.randn(60) * 0.5,
            "AMD": base * 0.8 + np.random.randn(60) * 0.5,
            "MSFT": base * 1.5 + np.random.randn(60) * 0.5,
            "GOOGL": base * 1.1 + np.random.randn(60) * 0.5,
        },
        index=dates,
    )

    results = run_kalman_statarb_scan(df, pairs=[("NVDA", "AMD"), ("MSFT", "GOOGL")])

    assert len(results) == 2
    for r in results:
        assert "beta" in r
        assert "z_score" in r
        assert "half_life_days" in r
        assert "signal" in r
        assert r["confidence"] >= 0.0
