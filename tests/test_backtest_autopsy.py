import numpy as np
import pandas as pd
import pytest
from src.backtest_autopsy import (
    compute_advanced_performance_ratios,
    compute_walk_forward_efficiency,
    run_autonomous_trade_autopsy,
)


def test_advanced_performance_ratios():
    # Synthetic upward equity series with one 10% drawdown
    equity = pd.Series([100.0, 105.0, 110.0, 100.0, 115.0, 125.0])
    ratios = compute_advanced_performance_ratios(equity)

    assert ratios["annualized_return_pct"] > 0.0
    assert ratios["max_drawdown_pct"] > 0.0
    assert ratios["calmar_ratio"] > 0.0
    assert ratios["sortino_ratio"] != 0.0
    assert ratios["omega_ratio"] > 0.0
    assert ratios["max_drawdown_duration_days"] >= 1


def test_walk_forward_efficiency():
    # Robust alpha: 2.5 IS Sharpe, 2.0 OOS Sharpe (80% WFE)
    res_robust = compute_walk_forward_efficiency(is_sharpe=2.5, oos_sharpe=2.0)
    assert res_robust["wfe_ratio"] == 0.80
    assert res_robust["wfe_pct"] == 80.0
    assert res_robust["classification"] == "EXCELLENT_ROBUST_ALPHA"

    # Overfitted alpha: 3.0 IS Sharpe, 0.9 OOS Sharpe (30% WFE)
    res_overfit = compute_walk_forward_efficiency(is_sharpe=3.0, oos_sharpe=0.9)
    assert res_overfit["wfe_ratio"] == 0.30
    assert res_overfit["classification"] == "OVERFITTING_WARNING"


def test_autonomous_trade_autopsy_mock(tmp_path):
    # Isolated mock trades CSV
    mock_csv = tmp_path / "mock_executed_trades.csv"
    df_mock = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "shares": 10,
                "entry_price": 100,
                "exit_price": 120,
                "pnl": 200.0,
                "return_pct": 20.0,
                "reason": "TP1_SCALE_OUT_50%",
            },
            {
                "ticker": "NVDA",
                "shares": 10,
                "entry_price": 100,
                "exit_price": 130,
                "pnl": 300.0,
                "return_pct": 30.0,
                "reason": "TP2_RUNNER_EXIT",
            },
            {
                "ticker": "AAPL",
                "shares": 5,
                "entry_price": 200,
                "exit_price": 190,
                "pnl": -50.0,
                "return_pct": -5.0,
                "reason": "STOP_LOSS",
            },
        ]
    )
    df_mock.to_csv(mock_csv, index=False)

    autopsy = run_autonomous_trade_autopsy(trades_path=str(mock_csv))
    assert autopsy["status"] == "AUTOPSY_COMPLETE"
    assert autopsy["total_trades"] == 3
    assert autopsy["winning_trades"] == 2
    assert autopsy["losing_trades"] == 1
    assert autopsy["win_rate_pct"] == 66.7
    assert autopsy["total_realized_pnl"] == 450.0
    assert autopsy["payoff_ratio"] == 5.0
    assert len(autopsy["episodic_lessons"]) >= 1


def test_autonomous_trade_autopsy_live_file_read_only():
    # Safely audit the real executed_trades.csv in read-only mode
    autopsy = run_autonomous_trade_autopsy()
    if autopsy.get("status") == "AUTOPSY_COMPLETE":
        assert autopsy["total_trades"] >= 40
        assert autopsy["win_rate_pct"] > 70.0
        assert autopsy["total_realized_pnl"] > 50_000.0
