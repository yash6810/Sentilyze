"""
Unit Tests for Autonomous Daily Execution Bridge (src/daily_execution_bridge.py).
STRICT PORTFOLIO PRESERVATION: Uses tmp_path for all portfolio files.
"""

import os
import json
import pytest
from src.daily_execution_bridge import DailyExecutionBridge


@pytest.fixture
def mock_signals_file(tmp_path):
    sig_path = tmp_path / "mock_signals.json"
    signals_data = {
        "scan_time": "2026-10-02T12:00:00Z",
        "num_assets": 4,
        "signals": [
            {
                "ticker": "AAPL",
                "signal": "BUY",
                "confidence": 0.72,
                "current_price": 220.0,
                "stop_loss": 210.0,
                "take_profit": 235.0,
            },
            {
                "ticker": "MSFT",
                "signal": "BUY",
                "confidence": 0.65,
                "current_price": 410.0,
                "stop_loss": 395.0,
                "take_profit": 435.0,
            },
            {
                "ticker": "GOOGL",
                "signal": "HOLD",
                "confidence": 0.50,
                "current_price": 175.0,
            },
            {
                "ticker": "TSLA",
                "signal": "SELL",
                "confidence": 0.35,
                "current_price": 240.0,
            },
        ],
    }
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(signals_data, f)
    return str(sig_path)


def test_bridge_categorization(mock_signals_file, tmp_path):
    p_file = str(tmp_path / "test_port.json")
    t_file = str(tmp_path / "test_trades.csv")

    bridge = DailyExecutionBridge(
        signals_file=mock_signals_file,
        portfolio_file=p_file,
        trades_file=t_file,
        dry_run=True,
    )
    categorized = bridge.get_actionable_signals(min_confidence=0.60)
    assert len(categorized["buys"]) == 2
    assert categorized["buys"][0]["ticker"] == "AAPL"
    assert categorized["buys"][1]["ticker"] == "MSFT"
    assert len(categorized["sells"]) == 1
    assert len(categorized["holds"]) == 1


def test_bridge_dry_run_execution(mock_signals_file, tmp_path):
    p_file = str(tmp_path / "test_port.json")
    t_file = str(tmp_path / "test_trades.csv")

    bridge = DailyExecutionBridge(
        signals_file=mock_signals_file,
        portfolio_file=p_file,
        trades_file=t_file,
        dry_run=True,
    )
    res = bridge.execute_daily_cycle()
    assert res["success"] is True
    assert res["dry_run"] is True
    assert "simulated_actions" in res
    # Portfolio equity should reflect capital minus minor realistic friction
    assert res["portfolio_state"]["total_equity"] >= 95000.0


def test_bridge_empty_signals(tmp_path):
    empty_file = str(tmp_path / "empty_signals.json")
    with open(empty_file, "w", encoding="utf-8") as f:
        json.dump({"signals": []}, f)

    p_file = str(tmp_path / "test_port.json")
    t_file = str(tmp_path / "test_trades.csv")

    bridge = DailyExecutionBridge(
        signals_file=empty_file,
        portfolio_file=p_file,
        trades_file=t_file,
        dry_run=True,
    )
    res = bridge.execute_daily_cycle()
    assert res["success"] is False
    assert res["reason"] == "NO_SIGNALS"
