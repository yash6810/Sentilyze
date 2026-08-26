import os
import json
import pytest
import pandas as pd
from src.paper_broker import PaperBroker


@pytest.fixture
def temp_portfolio_file(tmp_path):
    portfolio_path = tmp_path / "test_paper_portfolio.json"
    return str(portfolio_path)


def test_paper_broker_initialization(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)
    summary = broker.get_portfolio_summary()

    assert summary["total_equity"] == 100000.0
    assert summary["cash"] == 100000.0
    assert summary["open_positions_count"] == 0
    assert summary["total_trades"] == 0
    assert summary["realized_pnl"] == 0.0


def test_paper_broker_execute_buy_signals(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    signals = [
        {
            "ticker": "AMD",
            "signal": "BUY",
            "confidence": 0.73,
            "current_price": 400.0,
            "take_profit": 450.0,
            "stop_loss": 370.0,
            "regime": "BULLISH",
        },
        {
            "ticker": "TSLA",
            "signal": "BUY",
            "confidence": 0.60,
            "current_price": 200.0,
            "take_profit": 230.0,
            "stop_loss": 185.0,
            "regime": "BULLISH",
        },
    ]

    actions = broker.execute_daily_signals(signals)

    assert len(actions["buys"]) == 2
    assert "AMD" in broker.state["open_positions"]
    assert "TSLA" in broker.state["open_positions"]
    assert broker.state["cash"] < 100000.0
    assert broker.state["total_equity"] == 100000.0

    summary = broker.get_portfolio_summary()
    assert summary["open_positions_count"] == 2


def test_paper_broker_take_profit_exit(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # 1. Buy position
    signals = [
        {
            "ticker": "META",
            "signal": "BUY",
            "confidence": 0.65,
            "current_price": 500.0,
            "take_profit": 550.0,
            "stop_loss": 470.0,
        }
    ]
    broker.execute_daily_signals(signals)
    assert "META" in broker.state["open_positions"]

    # 2. Next day price jumps past TP1 target ($560 >= $550) -> 50% Scale-Out
    next_day_signals = [
        {
            "ticker": "META",
            "signal": "BUY",
            "confidence": 0.65,
            "current_price": 560.0,
            "take_profit": 600.0,
            "stop_loss": 520.0,
        }
    ]
    actions = broker.execute_daily_signals(next_day_signals)

    assert len(actions["take_profits"]) == 1
    assert actions["take_profits"][0]["reason"] == "TAKE_PROFIT"
    assert broker.state["open_positions"]["META"]["scaled_out"] is True
    assert broker.state["realized_pnl"] > 0

    # 3. Day 3 price reaches TP2 runner target ($600 >= $590) -> Complete Exit
    day3_signals = [
        {
            "ticker": "META",
            "signal": "HOLD",
            "confidence": 0.50,
            "current_price": 600.0,
            "take_profit": 650.0,
            "stop_loss": 550.0,
        }
    ]
    actions3 = broker.execute_daily_signals(day3_signals)
    assert "META" not in broker.state["open_positions"]
    assert broker.state["winning_trades"] >= 1


def test_paper_broker_stop_loss_exit(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # 1. Buy position
    signals = [
        {
            "ticker": "NVDA",
            "signal": "BUY",
            "confidence": 0.55,
            "current_price": 200.0,
            "take_profit": 230.0,
            "stop_loss": 180.0,
        }
    ]
    broker.execute_daily_signals(signals)

    # 2. Next day price drops below Stop-Loss ($175 <= $180)
    next_day_signals = [
        {
            "ticker": "NVDA",
            "signal": "BUY",
            "confidence": 0.55,
            "current_price": 175.0,
            "take_profit": 210.0,
            "stop_loss": 160.0,
        }
    ]
    actions = broker.execute_daily_signals(next_day_signals)

    assert len(actions["stop_losses"]) == 1
    assert actions["stop_losses"][0]["reason"] == "STOP_LOSS"
    assert "NVDA" not in broker.state["open_positions"]
    assert broker.state["realized_pnl"] < 0
    assert broker.state["losing_trades"] == 1


def test_paper_broker_dataframes(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)
    signals = [
        {
            "ticker": "AAPL",
            "signal": "BUY",
            "confidence": 0.58,
            "current_price": 300.0,
            "take_profit": 330.0,
            "stop_loss": 280.0,
            "regime": "BULLISH",
        }
    ]
    broker.execute_daily_signals(signals)

    pos_df = broker.get_open_positions_df()
    assert not pos_df.empty
    assert "AAPL" in pos_df["Ticker"].values

    eq_df = broker.get_equity_curve_df()
    assert not eq_df.empty
    assert "total_equity" in eq_df.columns
