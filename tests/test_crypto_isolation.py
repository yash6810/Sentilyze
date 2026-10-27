"""
Tests for strict Crypto Asset Isolation and US Market Hours Gating.
Verifies:
1. When US market is closed, cycle skips immediately (no scanning, zero Alpaca calls, zero funnel logs).
2. A crypto buy on the local ledger gives zero submit_bracket_order calls.
3. A stock buy still calls submit_bracket_order once.
4. AlpacaBrokerBridge directly rejects crypto tickers for submit_bracket_order and close_position.
"""

import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.autonomous_trader import AutonomousTradingEngine, LOCK_FILE
from src.paper_broker import PaperBroker
from src.alpaca_broker import AlpacaBrokerBridge


@pytest.fixture(autouse=True)
def clean_lock_file():
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass
    yield
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass


@patch("src.market_session.get_us_market_session")
@patch("src.autonomous_trader.fetch_universe_live_quotes")
@patch("src.autonomous_trader.convene_trading_committee")
@patch("src.alpaca_broker.AlpacaBrokerBridge")
@patch("src.funnel_tracker.record_funnel_run")
def test_market_closed_gives_no_scan_and_zero_alpaca_calls(
    mock_record_funnel, mock_alpaca, mock_committee, mock_quotes, mock_session
):
    """(a) When US market is closed, exit cycle immediately: no scan, zero Alpaca calls, no funnel row."""
    mock_session.return_value = {
        "status": "MARKET_CLOSED_WEEKEND",
        "is_open": False,
        "time_edt": "10:00:00 AM EDT",
        "utc_time": "2026-10-27 14:00:00 UTC",
    }

    mock_broker = MagicMock(spec=PaperBroker)
    mock_broker.state = {"cash": 100000.0, "open_positions": {}}
    mock_broker.get_portfolio_summary.return_value = {
        "total_equity": 100000.0,
        "cash": 100000.0,
        "unrealized_pnl": 0.0,
    }

    engine = AutonomousTradingEngine(broker=mock_broker)
    res = engine.run_autonomous_cycle(candidate_tickers=["AAPL", "BTC-USD"])

    assert res["status"] == "MARKET_CLOSED_SKIP"
    assert mock_quotes.called is False
    assert mock_committee.called is False
    assert mock_record_funnel.called is False
    assert mock_alpaca.return_value.submit_bracket_order.called is False


@patch("src.alpaca_broker.requests.post")
def test_alpaca_broker_crypto_guard_blocks_orders_and_closes(mock_post):
    """Direct AlpacaBrokerBridge policy verification: crypto tickers are blocked before network calls."""
    alpaca = AlpacaBrokerBridge()
    # 1. submit_bracket_order for crypto
    res_order = alpaca.submit_bracket_order(
        ticker="BTC-USD",
        qty=1,
        take_profit_price=70000.0,
        stop_loss_price=65000.0,
        side="buy",
    )
    assert res_order["status"] == "BLOCKED_CRYPTO_ASSET"
    assert mock_post.called is False

    # 2. close_position for crypto
    res_close = alpaca.close_position("ETH-USD")
    assert res_close["status"] == "BLOCKED_CRYPTO_ASSET"
    assert mock_post.called is False


@patch("src.market_session.get_us_market_session")
@patch("src.autonomous_trader.fetch_universe_live_quotes")
@patch("src.autonomous_trader.get_news")
@patch("src.autonomous_trader.convene_trading_committee")
@patch("src.alpaca_broker.AlpacaBrokerBridge")
def test_crypto_buy_on_local_ledger_gives_zero_submit_bracket_order_calls(
    mock_alpaca_cls, mock_committee, mock_news, mock_quotes, mock_session
):
    """(b) A crypto buy on the local ledger gives zero submit_bracket_order calls."""
    mock_session.return_value = {
        "status": "REGULAR_MARKET_OPEN",
        "is_open": True,
        "time_edt": "10:00:00 AM EDT",
        "utc_time": "2026-10-27 14:00:00 UTC",
    }
    mock_quotes.return_value = {
        "BTC-USD": {"ticker": "BTC-USD", "price": 68000.0, "status": "LIVE"}
    }
    mock_news.return_value = MagicMock()
    mock_committee.return_value = {
        "ticker": "BTC-USD",
        "spot_price": 68000.0,
        "final_resolution": "🚀 HIGH CONVICTION COMMITTEE BUY",
        "action_code": "EXECUTE_BUY",
        "tp1_target": 72000.0,
        "stop_loss_target": 65000.0,
        "cro_signoff": {
            "approved_leverage": 1.0,
            "approved_kelly_pct": 5.0,
        },
    }

    mock_alpaca_instance = MagicMock()
    mock_alpaca_instance.is_connected.return_value = True
    mock_alpaca_cls.return_value = mock_alpaca_instance

    mock_broker = MagicMock(spec=PaperBroker)
    mock_broker.state = {
        "cash": 150000.0,
        "open_positions": {},
        "closed_trades": [],
        "total_equity": 150000.0,
    }
    mock_broker.get_portfolio_summary.return_value = {
        "total_equity": 150000.0,
        "cash": 150000.0,
        "unrealized_pnl": 0.0,
    }

    engine = AutonomousTradingEngine(broker=mock_broker)
    res = engine.run_autonomous_cycle(candidate_tickers=["BTC-USD"])

    # Crypto ticker was excluded from candidate scanning or guarded from Alpaca
    assert mock_alpaca_instance.submit_bracket_order.called is False


@patch("src.data_ingestion.get_price_history")
@patch("src.market_session.get_us_market_session")
@patch("src.autonomous_trader.fetch_universe_live_quotes")
@patch("src.autonomous_trader.get_news")
@patch("src.autonomous_trader.convene_trading_committee")
@patch("src.alpaca_broker.AlpacaBrokerBridge")
def test_stock_buy_still_calls_submit_bracket_order_once(
    mock_alpaca_cls, mock_committee, mock_news, mock_quotes, mock_session, mock_hist
):
    """(c) A stock buy still calls submit_bracket_order once."""
    mock_session.return_value = {
        "status": "REGULAR_MARKET_OPEN",
        "is_open": True,
        "time_edt": "10:00:00 AM EDT",
        "utc_time": "2026-10-27 14:00:00 UTC",
    }
    # Provide bullish SPY data so Macro Breadth Gate is open
    spy_series = pd.Series([500.0] * 60 + [550.0])
    mock_hist.return_value = pd.DataFrame({"Close": spy_series})

    mock_quotes.return_value = {
        "NVDA": {
            "ticker": "NVDA",
            "price": 125.0,
            "status": "LIVE",
            "volume": 1000000,
            "change_pct": 2.5,
        },
        "^VIX": {"price": 15.0},
    }
    mock_news.return_value = MagicMock()
    mock_committee.return_value = {
        "ticker": "NVDA",
        "spot_price": 125.0,
        "final_resolution": "🚀 HIGH CONVICTION COMMITTEE BUY",
        "action_code": "EXECUTE_BUY",
        "tp1_target": 132.0,
        "stop_loss_target": 119.0,
        "cro_signoff": {
            "approved_leverage": 1.0,
            "approved_kelly_pct": 5.0,
        },
    }

    mock_alpaca_instance = MagicMock()
    mock_alpaca_instance.is_connected.return_value = True
    mock_alpaca_instance.get_open_positions.return_value = []
    mock_alpaca_instance.get_open_orders.return_value = []
    mock_alpaca_instance.get_account_summary.return_value = {
        "status": "CONNECTED",
        "equity": 100000.0,
        "cash": 100000.0,
    }
    mock_alpaca_cls.return_value = mock_alpaca_instance

    mock_broker = MagicMock(spec=PaperBroker)
    mock_broker.state = {
        "cash": 150000.0,
        "open_positions": {},
        "closed_trades": [],
        "total_equity": 150000.0,
    }
    mock_broker.get_portfolio_summary.return_value = {
        "total_equity": 150000.0,
        "cash": 150000.0,
        "unrealized_pnl": 0.0,
    }
    mock_broker.execute_buy.return_value = {
        "success": True,
        "shares": 40,
        "price": 125.0,
        "ticker": "NVDA",
    }

    engine = AutonomousTradingEngine(broker=mock_broker)
    res = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])

    assert mock_alpaca_instance.submit_bracket_order.call_count == 1
    call_args = mock_alpaca_instance.submit_bracket_order.call_args[1]
    assert call_args["ticker"] == "NVDA"
    assert call_args["qty"] == 40
    assert call_args["side"] == "buy"
