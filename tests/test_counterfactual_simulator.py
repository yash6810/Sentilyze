"""
Unit tests for Counterfactual Trade Replay Simulator (Sprint 1, Module 1.4)
"""

import os
import pytest
import pandas as pd
from src.counterfactual_simulator import CounterfactualTradeSimulator


@pytest.fixture
def sample_trades_csv(tmp_path):
    csv_file = tmp_path / "mock_executed_trades.csv"
    data = [
        {
            "ticker": "NVDA",
            "shares": 10,
            "entry_price": 100.0,
            "exit_price": 110.0,
            "pnl": 100.0,
            "return_pct": 10.0,
            "reason": "TP1",
        },
        {
            "ticker": "AAPL",
            "shares": 20,
            "entry_price": 150.0,
            "exit_price": 145.0,
            "pnl": -100.0,
            "return_pct": -3.33,
            "reason": "SL",
        },
        {
            "ticker": "MSFT",
            "shares": 5,
            "entry_price": 300.0,
            "exit_price": 315.0,
            "pnl": 75.0,
            "return_pct": 5.0,
            "reason": "TP1",
        },
        {
            "ticker": "AMZN",
            "shares": 15,
            "entry_price": 120.0,
            "exit_price": 118.0,
            "pnl": -30.0,
            "return_pct": -1.67,
            "reason": "SL",
        },
    ]
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_load_trades(sample_trades_csv):
    sim = CounterfactualTradeSimulator(trades_path=sample_trades_csv)
    df = sim.load_trades()
    assert not df.empty
    assert len(df) == 4
    assert "ticker" in df.columns
    assert "pnl" in df.columns


def test_simulate_single_trade():
    sim = CounterfactualTradeSimulator()
    trade = pd.Series(
        {
            "ticker": "NVDA",
            "shares": 10,
            "entry_price": 100.0,
            "exit_price": 110.0,
            "pnl": 100.0,
            "return_pct": 10.0,
        }
    )
    res = sim.simulate_single_trade(trade, stop_multiplier=2.0, tp_pct=0.05)
    assert res["ticker"] == "NVDA"
    assert "cf_pnl" in res
    assert "cf_ret_pct" in res
    assert "pnl_delta" in res


def test_simulate_grid(sample_trades_csv):
    sim = CounterfactualTradeSimulator(trades_path=sample_trades_csv)
    grid = sim.simulate_grid(stop_multipliers=[1.5, 2.5], tp_pcts=[0.03, 0.07])
    assert grid["status"] == "SUCCESS"
    assert grid["trades_count"] == 4
    assert len(grid["grid_results"]) == 4
    assert "optimal_policy" in grid
    assert "profit_factor" in grid["optimal_policy"]


def test_generate_report(sample_trades_csv, tmp_path):
    report_file = tmp_path / "mock_report.json"
    sim = CounterfactualTradeSimulator(trades_path=sample_trades_csv)
    rep = sim.generate_report(output_path=str(report_file))
    assert rep["module"].startswith("Sprint 1.4")
    assert os.path.exists(str(report_file))
