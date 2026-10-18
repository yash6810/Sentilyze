"""
Unit Tests for Soros Reflexivity & Dynamic Sympathy Lead-Lag Propagator.
Validates:
1. Soros kinematic phase-space calculations and regime classification.
2. Stealth Accumulation detection (+1.35x conviction boost).
3. Distribution Exhaustion detection (CRO veto).
4. Sympathy impulse calculation and follower lag opportunity detection.
"""

import numpy as np
import pandas as pd
import pytest

from src.soros_sympathy import (
    SorosReflexivityEngine,
    SympathyLeadLagPropagator,
    get_soros_engine,
    get_sympathy_propagator,
)


def test_soros_stealth_accumulation():
    """Verify that accelerating sentiment with flat price triggers STEALTH_ACCUMULATION."""
    engine = SorosReflexivityEngine(step_k=2)
    # Price is flat/pinned
    prices = np.array([100.0, 100.1, 100.0, 100.0, 100.05, 100.02])
    # Sentiment accelerates rapidly from neutral to positive
    sentiments = np.array([0.0, 0.05, 0.15, 0.35, 0.65, 0.95])

    result = engine.calculate_kinematics(prices, sentiments)
    assert result["regime"] == "STEALTH_ACCUMULATION"
    assert result["conviction_boost"] >= 1.25
    assert result["cro_veto"] is False


def test_soros_distribution_exhaustion():
    """Verify that peak sentiment with decelerating price triggers DISTRIBUTION_EXHAUSTION."""
    engine = SorosReflexivityEngine(step_k=2)
    # Price was surging, now stalling / decelerating
    prices = np.array([90.0, 95.0, 100.0, 105.0, 106.0, 105.5])
    # Sentiment is at peak retail euphoria (> 0.75) and plateauing
    sentiments = np.array([0.60, 0.70, 0.80, 0.85, 0.88, 0.88])

    result = engine.calculate_kinematics(prices, sentiments)
    assert result["regime"] == "DISTRIBUTION_EXHAUSTION"
    assert result["cro_veto"] is True
    assert result["conviction_boost"] == 0.0


def test_sympathy_driver_impulse_and_follower_lag():
    """Verify that a surging driver (NVDA) creates a sympathy opportunity on follower (AMD)."""
    propagator = SympathyLeadLagPropagator()

    # Create driver DataFrame (NVDA surging +3.5% on 2x volume)
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    driver_prices = np.linspace(120.0, 130.0, 20)
    driver_prices[-1] = 135.0  # Big jump
    driver_vol = np.full(20, 1000000)
    driver_vol[-1] = 2500000  # High volume
    driver_df = pd.DataFrame(
        {"Close": driver_prices, "Volume": driver_vol}, index=dates
    )

    # Create follower DataFrame (AMD flat, hasn't moved yet)
    follower_prices = np.linspace(150.0, 151.0, 20)
    follower_df = pd.DataFrame(
        {"Close": follower_prices, "Volume": np.full(20, 500000)}, index=dates
    )

    eval_result = propagator.evaluate_sympathy_for_follower(
        follower_ticker="AMD",
        driver_data_dict={"NVDA": driver_df},
        follower_df=follower_df,
        driver_sentiment_dict={"NVDA": 0.90},
    )

    assert eval_result["follower_ticker"] == "AMD"
    assert len(eval_result["active_drivers"]) >= 1
    assert eval_result["total_expected_sympathy_pct"] > 1.0
    assert eval_result["sympathy_opportunity"] is True
    assert eval_result["action"] == "FRONTRUN_SYMPATHY_WAVE"


def test_calculate_transfer_entropy():
    """Verify Invention 3: Non-linear Transfer Entropy detects causal lead-lag flow."""
    from src.soros_sympathy import calculate_transfer_entropy

    np.random.seed(42)
    # 1. Independent series: Transfer entropy should be near 0 with sufficient sample
    x_rand = np.random.randn(400)
    y_rand = np.random.randn(400)
    te_noise = calculate_transfer_entropy(x_rand, y_rand, bins=4)
    assert te_noise < 0.10

    # 2. Causal lead-lag series: Y_t is driven by X_{t-1}
    x_driver = np.random.randn(400)
    y_follower = np.zeros(400)
    for t in range(1, 400):
        y_follower[t] = 0.85 * x_driver[t - 1] + 0.15 * np.random.randn()

    te_causal = calculate_transfer_entropy(x_driver, y_follower, bins=4)
    te_reverse = calculate_transfer_entropy(y_follower, x_driver, bins=4)

    assert te_causal > 0.50
    assert te_causal > 5.0 * te_reverse
