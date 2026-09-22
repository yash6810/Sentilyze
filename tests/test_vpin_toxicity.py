"""
Unit tests for Volume-Synchronized Probability of Toxicity (VPIN) Microstructure Engine.
Tests Bulk Volume Classification (BVC), volume bucket partitioning, and CRO veto flags.
"""

import pytest
import numpy as np
import pandas as pd

from src.vpin_toxicity import (
    calculate_vpin,
    evaluate_ticker_toxicity,
)


def test_vpin_with_insufficient_data():
    df_empty = pd.DataFrame()
    res = calculate_vpin(df_empty)
    assert res["vpin"] == 0.35
    assert res["is_toxic_dumping"] is False
    assert res["cro_veto_recommended"] is False


def test_vpin_bounded_between_zero_and_one():
    # Generate synthetic price and volume series
    np.random.seed(42)
    n = 100
    prices = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    volumes = np.random.uniform(1000, 5000, n)
    df = pd.DataFrame({"Close": prices, "Volume": volumes})

    res = calculate_vpin(df, num_buckets=20, rolling_window=10)

    assert "vpin" in res
    assert 0.0 <= res["vpin"] <= 1.0
    assert "toxicity_regime" in res
    assert "is_toxic_dumping" in res
    assert "cro_veto_recommended" in res
    assert res["bucket_size"] > 0


def test_vpin_detects_severe_toxic_dumping():
    # Simulate a toxic market maker dump: severe continuous downward price bars on huge volume
    n = 100
    # Continuous severe drops
    price_drops = -np.abs(np.random.normal(3.0, 0.5, n))
    prices = 200.0 + np.cumsum(price_drops)
    # Huge volume during dumping
    volumes = np.full(n, 50000.0)
    df_toxic = pd.DataFrame({"Close": prices, "Volume": volumes})

    res = calculate_vpin(df_toxic, num_buckets=20, rolling_window=10)

    # In a pure one-sided dump, order imbalance approaches 1.0 (very high VPIN)
    assert res["vpin"] >= 0.70
    assert res["toxicity_regime"] in [
        "TOXIC_INSTITUTIONAL_DUMPING",
        "ELEVATED_INVENTORY_RISK",
    ]


def test_vpin_ticker_fallback():
    # Non-existent ticker fallback
    res = evaluate_ticker_toxicity("INVALID_TICKER_XYZ_123")
    assert "vpin" in res
    assert 0.0 <= res["vpin"] <= 1.0
    assert res["cro_veto_recommended"] is False
