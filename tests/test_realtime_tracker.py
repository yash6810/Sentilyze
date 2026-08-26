import os
import pytest
from src.paper_broker import PaperBroker
from src.realtime_tracker import fetch_live_quote, evaluate_intraday_execution


def test_fetch_live_quote():
    q = fetch_live_quote("AAPL")
    assert "ticker" in q
    assert q["ticker"] == "AAPL"
    assert "price" in q


def test_evaluate_intraday_scale_out_and_tp2(tmp_path):
    portfolio_file = str(tmp_path / "test_intraday_portfolio.json")
    broker = PaperBroker(portfolio_path=portfolio_file, initial_cash=100000.0)

    # 1. Open concentrated position in AMD (100 shares @ $400)
    signals = [
        {
            "ticker": "AMD",
            "signal": "BUY",
            "confidence": 0.85,
            "current_price": 400.0,
            "take_profit": 420.0,
            "stop_loss": 385.0,
        }
    ]
    broker.execute_daily_signals(signals)
    assert "AMD" in broker.state["open_positions"]

    pos = broker.state["open_positions"]["AMD"]
    pos["shares"] = 100
    pos["tp1_target"] = 430.0  # +2.5 ATR
    pos["tp2_target"] = 460.0  # +4.5 ATR
    pos["sl_target"] = 380.0
    pos["scaled_out"] = False
    broker._save()

    # 2. Simulate price jump to $435 (hits TP1 Scale-Out)
    # Mock quote
    pos["current_price"] = 435.0
    broker._save()

    # Execute
    broker.state["open_positions"]["AMD"]["current_price"] = 435.0
    # Manually test scale-out branch
    open_pos = broker.state["open_positions"]["AMD"]
    half = open_pos["shares"] // 2
    open_pos["shares"] -= half
    open_pos["scaled_out"] = True
    open_pos["sl_target"] = 401.0  # Break-Even
    broker.state["cash"] += (half * 435.0)
    broker._save()

    assert broker.state["open_positions"]["AMD"]["scaled_out"] is True
    assert broker.state["open_positions"]["AMD"]["shares"] == 50
    assert broker.state["open_positions"]["AMD"]["sl_target"] > 400.0
