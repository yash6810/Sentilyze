import pytest
import pandas as pd
from src.portfolio import (
    calculate_risk_parity_weights,
    build_unified_portfolio,
)


@pytest.fixture
def sample_portfolio_returns():
    dates = pd.date_range("2025-01-01", periods=10)
    returns_df = pd.DataFrame(
        {
            "NVDA": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.01, -0.03, 0.02, 0.01],
            "MSFT": [
                0.005,
                -0.002,
                0.008,
                0.003,
                -0.004,
                0.006,
                0.002,
                -0.005,
                0.003,
                0.001,
            ],
        },
        index=dates,
    )
    return returns_df


def test_calculate_risk_parity_weights(sample_portfolio_returns):
    weights = calculate_risk_parity_weights(sample_portfolio_returns)
    assert len(weights) == 2
    assert weights.sum() == pytest.approx(1.0)
    # Lower volatility asset (MSFT) should have a higher weight than NVDA
    assert weights["MSFT"] > weights["NVDA"]


def test_build_unified_portfolio(tmp_path):
    dates = pd.date_range("2025-01-01", periods=10)

    # Create fake portfolio CSVs in tmp_path
    for ticker in ["AAPL", "MSFT"]:
        df = pd.DataFrame(
            {
                "total": [10000.0 * (1.01**i) for i in range(10)],
                "benchmark": [10000.0 * (1.005**i) for i in range(10)],
                "cash": [1000.0] * 10,
                "holdings": [9000.0] * 10,
                "signal": [1] * 10,
            },
            index=dates,
        )
        df.to_csv(tmp_path / f"{ticker}_portfolio.csv")

    unified_df, metrics, weights_df = build_unified_portfolio(
        initial_capital=100000.0,
        results_dir=str(tmp_path),
        tickers=["AAPL", "MSFT"],
        allocation_method="risk_parity",
    )

    assert isinstance(unified_df, pd.DataFrame)
    assert len(unified_df) == 10
    assert "total" in unified_df.columns
    assert "benchmark" in unified_df.columns
    assert metrics["final_value"] > 100000.0
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert len(weights_df) == 2
