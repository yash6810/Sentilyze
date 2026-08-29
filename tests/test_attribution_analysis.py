import pytest
import pandas as pd
import numpy as np
from src.attribution_analysis import (
    run_attribution_decomposition,
    _generate_surrogate_ml_probabilities,
)


def test_surrogate_ml_probabilities():
    dates = pd.date_range("2022-01-01", periods=150, freq="B")
    prices = np.cumprod(1 + np.random.normal(0.001, 0.02, size=len(dates))) * 100.0
    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(100000, 500000, size=len(dates)),
        },
        index=dates,
    )

    probs = _generate_surrogate_ml_probabilities(df)
    assert len(probs) == len(df)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_attribution_decomposition(tmp_path, mocker):
    mocker.patch(
        "src.attribution_analysis.ATTRIBUTION_RESULTS_FILE",
        str(tmp_path / "test_attribution.json"),
    )

    # Mock price history
    dates = pd.date_range("2021-01-01", periods=250, freq="B")
    prices = np.cumprod(1 + np.random.normal(0.001, 0.015, size=len(dates))) * 100.0
    mock_df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(100000, 500000, size=len(dates)),
        },
        index=dates,
    )
    mocker.patch("src.attribution_analysis.get_price_history", return_value=mock_df)

    res = run_attribution_decomposition(
        ticker="NVDA", initial_capital=10000.0, n_random_trials=5, seed=42
    )

    assert res["ticker"] == "NVDA"
    assert "models" in res
    assert "full_ml_strategy" in res["models"]
    assert "always_long_strategy" in res["models"]
    assert "random_signal_strategy" in res["models"]
    assert "buy_and_hold_benchmark" in res["models"]
    assert "attribution_decomposition" in res
    assert "ml_predictive_edge_share_pct" in res["attribution_decomposition"]
    assert "risk_trade_management_share_pct" in res["attribution_decomposition"]
