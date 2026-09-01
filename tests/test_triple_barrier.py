import pytest
import numpy as np
import pandas as pd
from src.triple_barrier import (
    apply_triple_barrier_labeling,
    calculate_deflated_sharpe_ratio,
)


def test_triple_barrier_labeling():
    dates = pd.date_range("2025-01-01", periods=30)
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 130, 30),
            "High": np.linspace(102, 132, 30),
            "Low": np.linspace(99, 129, 30),
            "Close": np.linspace(101, 131, 30),
        },
        index=dates,
    )

    res = apply_triple_barrier_labeling(
        df, profit_taking_mult=2.0, stop_loss_mult=1.5, max_holding_days=5
    )
    assert "target_barrier" in res.columns
    assert "barrier_return" in res.columns
    assert set(res["target_barrier"].unique()).issubset({-1, 0, 1})


def test_deflated_sharpe_ratio():
    np.random.seed(42)
    # Generate positive synthetic strategy returns
    rets = pd.Series(np.random.normal(0.001, 0.01, 252))
    dsr = calculate_deflated_sharpe_ratio(rets, num_trials=20)
    assert "annualized_sharpe" in dsr
    assert "dsr_probability" in dsr
    assert 0.0 <= dsr["dsr_probability"] <= 1.0
