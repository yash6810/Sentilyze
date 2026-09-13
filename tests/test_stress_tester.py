"""
Unit tests for Stress Testing & Black Swan Crisis Simulation Suite.
"""

import json
import pytest
from src.stress_tester import (
    run_monte_carlo_stress_test,
    run_monte_carlo_var,
    run_full_crisis_simulation_suite,
)


def test_monte_carlo_stress_test_basic():
    res = run_monte_carlo_stress_test(
        initial_capital=100000.0,
        num_simulations=500,
        time_horizon_days=20,
    )
    assert res["initial_capital"] == 100000.0
    assert res["time_horizon_days"] == 20
    assert res["num_simulations"] == 500
    assert res["var_95_dollar"] > 0
    assert res["var_95_pct"] > 0
    assert "percentile_paths_df" in res


def test_monte_carlo_var_wrapper():
    res = run_monte_carlo_var(initial_equity=50000.0, num_paths=200, days=15)
    assert res["initial_capital"] == 50000.0
    assert "prob_profit_pct" in res


def test_run_full_crisis_simulation_suite(tmp_path):
    dummy_portfolio = tmp_path / "dummy_portfolio.json"
    dummy_output = tmp_path / "crisis_audit.json"

    port_data = {
        "total_equity": 125000.0,
        "cash": 50000.0,
        "open_positions": {
            "NVDA": {
                "shares": 50,
                "entry_price": 120.0,
                "current_price": 130.0,
            },
            "AAPL": {
                "shares": 100,
                "entry_price": 200.0,
                "current_price": 210.0,
            },
        },
    }
    with open(dummy_portfolio, "w", encoding="utf-8") as f:
        json.dump(port_data, f)

    audit = run_full_crisis_simulation_suite(
        portfolio_path=str(dummy_portfolio),
        output_path=str(dummy_output),
    )

    assert audit["status"] == "PASS"
    assert audit["portfolio_equity"] == 125000.0
    assert audit["cash"] == 50000.0
    assert audit["open_positions_count"] == 2
    assert len(audit["crisis_simulations"]) >= 4
    assert "monte_carlo_metrics" in audit
    assert dummy_output.exists()


def test_run_full_crisis_simulation_suite_empty_positions(tmp_path):
    dummy_portfolio = tmp_path / "empty_portfolio.json"
    dummy_output = tmp_path / "empty_crisis_audit.json"

    port_data = {
        "total_equity": 150000.0,
        "cash": 150000.0,
        "open_positions": {},
    }
    with open(dummy_portfolio, "w", encoding="utf-8") as f:
        json.dump(port_data, f)

    audit = run_full_crisis_simulation_suite(
        portfolio_path=str(dummy_portfolio),
        output_path=str(dummy_output),
    )

    assert audit["status"] == "PASS"
    assert audit["portfolio_equity"] == 150000.0
    assert audit["cash"] == 150000.0
    assert audit["open_positions_count"] == 0
    assert audit["invested_capital"] == 0.0
    assert dummy_output.exists()
