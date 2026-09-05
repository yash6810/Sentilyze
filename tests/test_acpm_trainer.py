"""
Unit Tests for 10x Alpha-Conformal Purged Multi-Task (ACPM) Training Engine.
"""

import os
import pytest
import numpy as np
import pandas as pd
import tempfile
import xgboost as xgb

from src.fractional_diff import (
    get_weights_ffd,
    fractional_differentiation_ffd,
    find_optimal_d,
)
from src.feature_neutralization import neutralize_features, neutralize_predictions
from src.cross_asset_pooling import get_sector_for_ticker, build_pooled_sector_dataset
from src.purged_cv import PurgedGroupTimeSeriesSplit, compute_deflated_sharpe_ratio
from src.regime_moe import RegimeMixtureOfExperts
from src.conformal_calibration import ConformalCalibrator
from src.acpm_trainer import ACPMTrainer


def test_fractional_differentiation():
    # Generate synthetic price series with trend
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    prices = pd.Series(
        100.0 + np.cumsum(np.random.randn(200)), index=dates, name="Close"
    )

    weights = get_weights_ffd(d=0.40)
    assert len(weights) > 0
    assert np.isclose(weights[-1], 1.0)

    ffd_series = fractional_differentiation_ffd(prices, d=0.40)
    assert len(ffd_series) > 0
    assert not ffd_series.isna().any()

    # Test optimal d finder
    best_d, best_series = find_optimal_d(prices)
    assert 0.0 < best_d <= 1.0
    assert len(best_series) > 0


def test_feature_neutralization():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "Feat1": np.random.randn(n),
            "Feat2": np.random.randn(n),
            "SPY_Return": np.random.randn(n),
        }
    )
    # Make Feat1 heavily correlated with SPY
    df["Feat1"] = df["Feat1"] + 3.0 * df["SPY_Return"]

    neutral_df = neutralize_features(
        df,
        target_columns=["Feat1", "Feat2"],
        factor_columns=["SPY_Return"],
        proportion=1.0,
    )

    # Verify linear correlation is stripped
    corr_before = np.corrcoef(df["Feat1"], df["SPY_Return"])[0, 1]
    corr_after = np.corrcoef(neutral_df["Feat1"], df["SPY_Return"])[0, 1]
    assert abs(corr_after) < abs(corr_before)
    assert abs(corr_after) < 1e-4


def test_cross_asset_pooling():
    assert get_sector_for_ticker("NVDA") == "Technology"
    assert get_sector_for_ticker("XOM") == "Energy"
    assert get_sector_for_ticker("JPM") == "Financials"

    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    df1 = pd.DataFrame(
        {"F1": np.random.randn(50), "Target": np.random.randint(0, 2, 50)}, index=dates
    )
    df2 = pd.DataFrame(
        {"F1": np.random.randn(50), "Target": np.random.randint(0, 2, 50)}, index=dates
    )

    ticker_dfs = {"NVDA": df1, "AAPL": df2}
    X_p, y_p = build_pooled_sector_dataset(
        ticker_dfs, feature_cols=["F1"], target_col="Target", sector="Technology"
    )
    assert len(X_p) == 100
    assert len(y_p) == 100
    assert "ticker_id" in X_p.columns


def test_purged_cv_and_dsr():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame({"F1": np.random.randn(n)}, index=dates)
    y = pd.Series(np.random.randint(0, 2, n), index=dates)

    cv = PurgedGroupTimeSeriesSplit(n_splits=3, purge_window=3, embargo_pct=0.05)
    splits = list(cv.split(X, y))
    assert len(splits) == 3

    for train_idx, test_idx in splits:
        # Assert no overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0

    dsr = compute_deflated_sharpe_ratio(
        estimated_sharpe=1.2,
        benchmark_sharpe=0.0,
        n_trials=20,
        var_sharpe=0.1,
        sample_length=252,
    )
    assert 0.0 <= dsr <= 1.0


def test_regime_moe():
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    X = pd.DataFrame(
        {
            "Close": 100 + np.cumsum(np.random.randn(n)),
            "SMA_50": 100 * np.ones(n),
            "RSI": np.random.uniform(30, 70, n),
            "MACD": np.random.randn(n),
        },
        index=dates,
    )
    y = pd.Series(np.random.randint(0, 2, n), index=dates)

    moe = RegimeMixtureOfExperts()
    moe.fit(X, y)
    probs = moe.predict_proba(X)
    preds = moe.predict(X)

    assert probs.shape == (n, 2)
    assert len(preds) == n
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_conformal_calibrator():
    np.random.seed(42)
    raw_scores = np.random.uniform(0.3, 0.7, 100)
    y_true = (raw_scores > 0.5).astype(int)

    calib = ConformalCalibrator(alpha=0.10)
    calib.fit(raw_scores, y_true)

    cal_scores = calib.calibrate(raw_scores)
    assert len(cal_scores) == 100
    assert np.all((cal_scores >= 0.0) & (cal_scores <= 1.0))
    # Monotonicity check
    sorted_idx = np.argsort(raw_scores)
    assert np.all(np.diff(cal_scores[sorted_idx]) >= -1e-6)


def test_acpm_trainer_end_to_end(tmp_path):
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "Close": 100 + np.cumsum(np.random.randn(n)),
            "SMA_50": 100 * np.ones(n),
            "RSI": np.random.uniform(30, 70, n),
            "MACD": np.random.randn(n),
            "Target": np.random.randint(0, 2, n),
        },
        index=dates,
    )

    spy = pd.Series(np.random.randn(n) * 0.01, index=dates)

    trainer = ACPMTrainer(n_splits=3, ffd_d=0.40, neutralize_beta=True, use_moe=True)
    model, metrics, oos_series = trainer.train_ticker(
        ticker="TEST_TICKER",
        df=df,
        feature_cols=["Close", "RSI", "MACD"],
        target_col="Target",
        benchmark_returns=spy,
    )

    assert model is not None
    assert "acpm_accuracy" in metrics
    assert "acpm_sharpe" in metrics
    assert "deflated_sharpe_ratio" in metrics
    assert len(oos_series) > 0
    assert os.path.exists("models/TEST_TICKER_model.json")

    # Cleanup test model
    if os.path.exists("models/TEST_TICKER_model.json"):
        os.remove("models/TEST_TICKER_model.json")
