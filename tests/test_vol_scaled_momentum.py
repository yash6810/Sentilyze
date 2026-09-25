"""
Unit Tests for Volatility-Scaled Multi-Horizon Momentum Engine (Daniel & Moskowitz 2016).
STRICT PORTFOLIO PRESERVATION: Uses synthetic in-memory data, zero impact on results/ files.
"""

import numpy as np
import pandas as pd
from src.vol_scaled_momentum import VolScaledMomentumEngine, VolScaledMomentumResult


def test_vol_scaled_momentum_synthetic_trending():
    engine = VolScaledMomentumEngine(target_vol=0.12, max_leverage=2.0)

    # Generate 300 days of upward trending price series
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=320, freq="B")
    trend = np.linspace(100, 200, 320)
    noise = np.random.normal(0, 1.5, 320)
    prices = trend + noise

    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 1000000,
        },
        index=dates,
    )

    res = engine.calculate_signal("TEST_BULL", df_history=df)
    assert isinstance(res, VolScaledMomentumResult)
    assert res.ticker == "TEST_BULL"
    # Upward trending price should yield a positive composite score and bullish regime
    assert res.composite_zscore > 0.0
    assert res.target_exposure > 0.0
    assert res.annualized_realized_vol > 0.0
    assert res.vol_scaling_multiplier > 0.0
    assert "BULLISH" in res.momentum_regime


def test_vol_scaled_momentum_insufficient_data():
    engine = VolScaledMomentumEngine()

    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    df = pd.DataFrame(
        {"Close": np.linspace(100, 110, 50)},
        index=dates,
    )

    res = engine.calculate_signal("SHORT_DATA", df_history=df)
    assert res.target_exposure == 0.0
    assert res.momentum_regime == "INSUFFICIENT_DATA"


def test_conditional_volatility_ewma():
    engine = VolScaledMomentumEngine(ewma_lambda=0.94)

    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.03, -0.025] * 20)
    vol = engine.compute_conditional_volatility(returns)
    assert len(vol) == len(returns)
    assert (vol > 0.0).all()
