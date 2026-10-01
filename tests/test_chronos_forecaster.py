"""
Unit tests for Chronos Foundation Forecaster & Price Cones (Sprint 3, Module 3.2)
"""

import pytest
import numpy as np
from src.models.chronos_forecaster import (
    ChronosTokenizer,
    ChronosPriceConeForecaster,
    get_chronos_price_forecast,
)


def test_chronos_tokenizer_encode_decode():
    tokenizer = ChronosTokenizer(n_bins=128)
    prices = np.array([100.0, 101.5, 102.0, 99.5, 103.0])
    tokens, scale = tokenizer.encode(prices)

    assert len(tokens) == len(prices)
    assert scale > 0.0
    assert (tokens >= 0).all() and (tokens < 128).all()

    recon = tokenizer.decode(tokens, scale, mean_val=np.mean(prices))
    assert len(recon) == len(prices)
    # Reasonable reconstruction correlation
    assert np.corrcoef(prices, recon)[0, 1] > 0.90


def test_chronos_sample_trajectories():
    forecaster = ChronosPriceConeForecaster()
    prices = np.linspace(100, 110, 30)
    samples = forecaster.sample_future_trajectories(
        prices, horizon_days=5, n_samples=20
    )
    assert samples.shape == (20, 5)
    assert (samples > 0).all()


def test_forecast_price_cones_smoke():
    res = get_chronos_price_forecast("NVDA", horizon=7)
    assert res["status"] == "SUCCESS"
    assert "quantile_cones" in res
    assert len(res["quantile_cones"]["q50_median"]) == 7
    assert len(res["quantile_cones"]["q10_floor"]) == 7
    # Cones should be ordered: q10 <= q50 <= q90
    for day in range(7):
        q10 = res["quantile_cones"]["q10_floor"][day]
        q50 = res["quantile_cones"]["q50_median"][day]
        q90 = res["quantile_cones"]["q90_ceiling"][day]
        assert q10 <= q50 <= q90
