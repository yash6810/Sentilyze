"""
Unit Tests for Conformalized Quantile Regression (CQR) Engine.
Validates distribution-free coverage, interval widths, and trade viability filtering.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.conformal_engine import (
    ConformalQuantileEngine,
    calibrate_and_evaluate_conformal_bounds,
)


@pytest.fixture
def synthetic_market_data():
    """Generates synthetic multi-feature time series and forward returns."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="D")

    # 5 synthetic technical & sentiment features
    X = pd.DataFrame(
        {
            "rsi": np.random.uniform(30, 70, n),
            "macd": np.random.normal(0, 1, n),
            "vpin_proxy": np.random.uniform(0.1, 0.6, n),
            "poc_distance_pct": np.random.normal(0, 0.02, n),
            "mean_sentiment_score": np.random.uniform(-0.5, 0.5, n),
        },
        index=dates,
    )

    # Target 1-day forward return with mild positive signal and heavy tails
    signal = 0.003 * (X["rsi"] > 50) + 0.005 * X["mean_sentiment_score"]
    noise = np.random.laplace(0, 0.015, n)
    y = pd.Series(signal + noise, index=dates, name="target_return_1d")

    return X, y


def test_conformal_initialization():
    engine = ConformalQuantileEngine(alpha=0.10)
    assert engine.alpha == 0.10
    assert engine.alpha_lo == 0.05
    assert engine.alpha_hi == 0.95
    assert engine.model_lo is None
    assert engine.conformal_quantile_margin == 0.0


def test_conformal_fit_and_calibrate(synthetic_market_data):
    X, y = synthetic_market_data
    engine = ConformalQuantileEngine(alpha=0.10, random_state=42)
    engine.fit_and_calibrate(X, y, calib_fraction=0.25)

    assert engine.model_lo is not None
    assert engine.model_hi is not None
    assert engine.calibration_size > 0
    assert isinstance(engine.conformal_quantile_margin, float)
    assert engine.mean_interval_width > 0.0


def test_conformal_prediction_output(synthetic_market_data):
    X, y = synthetic_market_data
    engine = ConformalQuantileEngine(alpha=0.10, random_state=42)
    engine.fit_and_calibrate(X, y)

    sample = X.iloc[[-1]]
    res = engine.predict_interval(sample)

    assert "lower_bound_pct" in res
    assert "upper_bound_pct" in res
    assert "interval_width_pct" in res
    assert "is_trade_viable" in res
    assert "signal_clarity" in res
    assert res["upper_bound_pct"] >= res["lower_bound_pct"]
    assert res["interval_width_pct"] > 0


def test_conformal_coverage_empirical(synthetic_market_data):
    """Verifies that out-of-sample test points are covered with expected frequency."""
    X, y = synthetic_market_data
    train_size = 100
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    engine = ConformalQuantileEngine(alpha=0.10, random_state=42)
    engine.fit_and_calibrate(X_train, y_train, calib_fraction=0.30)

    covered = 0
    total = len(X_test)
    for i in range(total):
        row = X_test.iloc[[i]]
        actual = y_test.iloc[i] * 100.0  # in pct
        pred = engine.predict_interval(row)
        if pred["lower_bound_pct"] <= actual <= pred["upper_bound_pct"]:
            covered += 1

    empirical_coverage = covered / total
    # Should achieve roughly ~80-95% coverage on test set
    assert empirical_coverage >= 0.70


def test_conformal_serialization(tmp_path, monkeypatch, synthetic_market_data):
    X, y = synthetic_market_data
    monkeypatch.setattr("src.conformal_engine.CONFORMAL_DIR", str(tmp_path))

    engine = ConformalQuantileEngine(alpha=0.10)
    engine.fit_and_calibrate(X, y)
    engine.save_calibration("TEST_TICKER")

    saved_file = tmp_path / "TEST_TICKER_cqr.json"
    assert saved_file.exists()

    meta = ConformalQuantileEngine.load_calibration_metadata("TEST_TICKER")
    assert meta is not None
    assert meta["ticker"] == "TEST_TICKER"
    assert meta["coverage_guarantee_pct"] == 90.0
