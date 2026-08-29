import pytest
import os
import pandas as pd
import numpy as np
from src.attribution_analysis import run_attribution_decomposition


def test_missing_predictions_file_raises_error():
    """Verify that attribution analysis strictly refuses to generate fake/surrogate data."""
    with pytest.raises(
        FileNotFoundError, match="FATAL: Real out-of-sample prediction file"
    ):
        run_attribution_decomposition(
            ticker="NONEXISTENT_TICKER_9999", n_random_trials=3
        )


def test_attribution_decomposition_nvda(tmp_path, mocker):
    mocker.patch(
        "src.attribution_analysis.ATTRIBUTION_RESULTS_FILE",
        str(tmp_path / "test_attribution.json"),
    )

    res = run_attribution_decomposition(
        ticker="NVDA", initial_capital=10000.0, n_random_trials=3, seed=42
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

    assert "ml_predictive_edge_share_pct" in res["attribution_decomposition"]
    assert "risk_trade_management_share_pct" in res["attribution_decomposition"]
