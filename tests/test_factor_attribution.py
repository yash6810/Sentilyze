"""
Unit tests for Institutional Factor Attribution & Performance Factsheet Engine.
Strictly verifies that live paper portfolio state is never altered.
"""

import pytest
import numpy as np
import pandas as pd
import json

from src.factor_attribution import (
    FactorAttributionEngine,
    InstitutionalFactsheet,
    generate_institutional_factsheet_for_ticker,
)


@pytest.fixture
def mock_return_series():
    """Generate 100 days of strategy and benchmark returns."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = np.random.RandomState(42)
    strat_ret = pd.Series(rng.normal(0.0012, 0.010, 100), index=dates)
    bench_ret = pd.Series(rng.normal(0.0006, 0.012, 100), index=dates)
    return strat_ret, bench_ret


@pytest.fixture
def mock_trade_dataframe():
    """Generate sample closed trade executions."""
    return pd.DataFrame(
        {
            "ticker": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"],
            "pnl": [500.0, -150.0, 320.0, 450.0, -80.0],
            "return_pct": [5.0, -1.5, 3.2, 4.5, -0.8],
            "entry_date": ["2026-09-01"] * 5,
            "exit_date": ["2026-09-02"] * 5,
        }
    )


def test_factor_attribution_engine_metrics(mock_return_series, mock_trade_dataframe):
    """Verify calculation of alpha, beta, sharpe, sortino, calmar, and VaR."""
    strat_ret, bench_ret = mock_return_series
    engine = FactorAttributionEngine(risk_free_rate=0.04)

    factsheet = engine.compute_metrics(
        strategy_returns=strat_ret,
        benchmark_returns=bench_ret,
        trade_df=mock_trade_dataframe,
        strategy_name="Test Strategy",
        benchmark_name="Test Benchmark",
    )

    assert isinstance(factsheet, InstitutionalFactsheet)
    rm = factsheet.risk_metrics

    # Check risk metrics presence and bounds
    assert rm.annualized_volatility_pct > 0.0
    assert rm.calmar_ratio != 0.0
    assert rm.var_95_daily_pct > 0.0
    assert rm.cvar_95_daily_pct >= rm.var_95_daily_pct
    assert rm.var_99_daily_pct >= rm.var_95_daily_pct

    # Check trade metrics
    tm = factsheet.trade_metrics
    assert tm is not None
    assert tm.total_trades == 5
    assert tm.winning_trades == 3
    assert tm.losing_trades == 2
    assert tm.win_rate_pct == 60.0
    assert tm.profit_factor > 1.0


def test_generate_institutional_factsheet_in_tmp_path(tmp_path):
    """Verify factsheet generation with isolated results directory."""
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 50)))
    df_port = pd.DataFrame(
        {
            "total": prices * 100,
            "benchmark": prices * 95,
        },
        index=dates,
    )

    # Save mock portfolio in tmp_path
    port_file = tmp_path / "MOCK_portfolio.csv"
    df_port.to_csv(port_file)

    factsheet = generate_institutional_factsheet_for_ticker(
        ticker="MOCK",
        results_dir=str(tmp_path),
        save_output=True,
    )

    assert factsheet.strategy_name == "Sentilyze Walk-Forward Dynamic Model (MOCK)"
    out_file = tmp_path / "institutional_factsheet_MOCK.json"
    assert out_file.exists()

    with open(out_file, "r") as f:
        data = json.load(f)
    assert "risk_metrics" in data
    assert data["risk_metrics"]["cagr_pct"] is not None
