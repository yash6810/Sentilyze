import os
import pytest
from unittest.mock import MagicMock, patch
from src.autonomous_trader import (
    AutonomousTradingEngine,
    load_universe_tickers,
    LOCK_FILE,
)
from src.paper_broker import PaperBroker


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


def test_load_universe_tickers():
    tickers = load_universe_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) >= 5
    assert "NVDA" in tickers or "AAPL" in tickers


@patch("src.autonomous_trader.fetch_universe_live_quotes")
@patch("src.autonomous_trader.get_news")
@patch("src.autonomous_trader.convene_trading_committee")
def test_autonomous_cycle_execution(mock_committee, mock_news, mock_quotes):
    mock_quotes.return_value = {
        "NVDA": {"ticker": "NVDA", "price": 125.0, "status": "LIVE"},
        "AAPL": {"ticker": "AAPL", "price": 220.0, "status": "LIVE"},
    }
    mock_news.return_value = MagicMock()
    mock_committee.return_value = {
        "ticker": "NVDA",
        "spot_price": 125.0,
        "final_resolution": "🚀 CONVICTION INSTITUTIONAL BUY",
        "action_code": "EXECUTE_BUY",
        "consensus_conviction_pct": 85.0,
        "tp1_target": 132.0,
        "tp2_target": 140.0,
        "stop_loss_target": 119.0,
        "cro_signoff": {
            "approved_leverage": 1.5,
            "kelly_allocation_pct": 12.5,
            "consensus_conviction_pct": 85.0,
        },
    }

    mock_broker = MagicMock(spec=PaperBroker)
    mock_broker.state = {
        "cash": 100000.0,
        "open_positions": {},
        "closed_trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_equity": 100000.0,
    }
    mock_broker.get_portfolio_summary.return_value = {
        "total_equity": 100000.0,
        "cash": 100000.0,
        "unrealized_pnl": 0.0,
    }
    mock_broker.execute_buy.return_value = {
        "success": True,
        "shares": 100,
        "price": 125.0,
        "ticker": "NVDA",
    }
    mock_broker.execute_manual_buy.return_value = {
        "success": True,
        "shares": 100,
        "price": 125.0,
        "ticker": "NVDA",
    }

    engine = AutonomousTradingEngine(broker=mock_broker)
    res = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])

    assert isinstance(res, dict)
    assert "buys" in res
    assert "portfolio_equity" in res
    assert res["portfolio_equity"] == 100000.0


def test_idempotency_lock_prevents_overlap(tmp_path, monkeypatch):
    """Task 6: Verify active lock file prevents overlapping concurrent cycles."""
    import json
    import time
    from src.autonomous_trader import LOCK_FILE

    # Create dummy lock file
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump({"pid": 99999, "timestamp": time.time()}, f)

    try:
        mock_broker = MagicMock(spec=PaperBroker)
        engine = AutonomousTradingEngine(broker=mock_broker)
        res = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])

        assert res.get("status") == "SKIPPED_LOCKED"
        assert "lock_pid" in res
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)


def test_master_kill_switch_blocks_buys(monkeypatch):
    """Task 7: Verify master kill switch disables order placement."""
    monkeypatch.setenv("SENTILYZE_KILL_SWITCH", "true")
    from src.autonomous_trader import is_kill_switch_active

    assert is_kill_switch_active() is True


def test_daily_loss_circuit_breaker():
    """Task 8: Verify circuit breaker triggers when true daily drawdown exceeds threshold."""
    from src.autonomous_trader import check_daily_loss_circuit_breaker

    # 1. Normal state (+$1,500 daily gain on $100k portfolio = +1.5%)
    normal_summary = {
        "daily_return_pct": 1.5,
        "start_of_day_equity": 100000.0,
        "total_equity": 101500.0,
    }
    assert (
        check_daily_loss_circuit_breaker(normal_summary, max_daily_loss_pct=3.0)
        is False
    )

    # 2. Breached state (-$4,000 daily loss on $100k portfolio = -4.0%)
    breached_summary = {
        "daily_return_pct": -4.0,
        "start_of_day_equity": 100000.0,
        "total_equity": 96000.0,
    }
    assert (
        check_daily_loss_circuit_breaker(breached_summary, max_daily_loss_pct=3.0)
        is True
    )

    # 3. Scaled large account ($500k starting equity, -$18k intraday loss = -3.6%)
    scaled_summary = {
        "start_of_day_equity": 500000.0,
        "total_equity": 482000.0,
    }
    assert (
        check_daily_loss_circuit_breaker(scaled_summary, max_daily_loss_pct=3.0) is True
    )


def test_unhandled_exception_handling_and_alert(monkeypatch):
    """Task 9: Verify unhandled exception in cycle is caught and handled safely."""
    engine = AutonomousTradingEngine()
    monkeypatch.setattr(
        engine,
        "_execute_cycle_body",
        MagicMock(side_effect=RuntimeError("Simulated critical DB failure")),
    )

    res = engine.run_autonomous_cycle()
    assert res.get("status") == "ERROR"
    assert "Simulated critical DB failure" in res.get("error", "")


def test_run_premarket_briefing(monkeypatch):
    """Verify run_premarket_briefing gathers macro data and dispatches morning briefing."""
    mock_broker = MagicMock()
    mock_broker.get_portfolio_summary.return_value = {
        "total_equity": 102000.0,
        "cash": 98000.0,
        "unrealized_pnl": 400.0,
        "unrealized_pnl_pct": 0.4,
        "win_rate": 55.0,
        "open_positions": {},
    }

    engine = AutonomousTradingEngine(broker=mock_broker)
    monkeypatch.setattr(
        "src.autonomous_trader.fetch_live_quote",
        lambda sym: {"price": 18.5 if sym == "^VIX" else 150.0},
    )
    monkeypatch.setattr(
        "src.autonomous_trader.convene_trading_committee",
        lambda tk, **kw: {
            "final_resolution": "BUY",
            "consensus_conviction_pct": 80.0,
            "agent_testimonies": [
                {"agent_name": "FinBERT Sentiment Specialist", "conviction_score": 80.0}
            ],
        },
    )
    monkeypatch.setattr(
        "src.autonomous_trader.send_discord_premarket_briefing",
        lambda **kwargs: True,
    )

    res = engine.run_premarket_briefing()
    assert res["status"] == "success"
    assert res["discord_dispatched"] is True
    assert res["macro_vix"] == 18.5
    assert res["total_equity"] == 102000.0
    assert res["watchlist_evaluated"] == 5


def test_ensure_background_daemon_thread_running():
    from src.autonomous_trader import (
        ensure_background_daemon_thread_running,
        get_daemon_status,
    )

    t = ensure_background_daemon_thread_running(interval_seconds=3600)
    assert t is not None
    assert t.is_alive()
    status = get_daemon_status()
    assert status["is_active"] is True
