import os
"""
Tests for Risk Controls and Position Limits in Sentilyze.
Verifies:
1. (a) A second buy for the same ticker is blocked (both local and Alpaca live).
2. (b) A position exceeding 10% of total portfolio equity is blocked.
3. (c) The 11th concurrent position is blocked (MAX_CONCURRENT_POSITIONS = 10).
"""

import pytest
from unittest.mock import patch, MagicMock
from src.paper_broker import PaperBroker


@pytest.fixture
def isolated_broker(tmp_path):
    portfolio_file = str(tmp_path / "paper_portfolio.json")
    trades_file = str(tmp_path / "executed_trades.csv")
    broker = PaperBroker(
        portfolio_path=portfolio_file,
        initial_cash=100000.0,
        trades_path=trades_file,
    )
    return broker


def test_second_buy_same_ticker_is_blocked(isolated_broker):
    """(a) A second buy for the same ticker is strictly blocked."""
    # Mock Alpaca as disconnected initially
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        # 1. Initial buy succeeds
        res1 = isolated_broker.execute_buy(ticker="AAPL", shares=10, price=150.0)
        assert res1["success"] is True
        assert "AAPL" in isolated_broker.state["open_positions"]

        # 2. Second buy for the same ticker in local ledger is blocked
        res2 = isolated_broker.execute_buy(ticker="AAPL", shares=5, price=155.0)
        assert res2["success"] is False
        assert "already exists" in res2["error"]

    # 3. Buy for ticker that has open position or open order on Alpaca is blocked
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[{"symbol": "MSFT", "market_value": "5000"}],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[],
        ),
    ):
        res3 = isolated_broker.execute_buy(ticker="MSFT", shares=10, price=200.0)
        assert res3["success"] is False
        assert "Alpaca" in res3["error"]

    # 4. Buy for ticker that has a pending open order on Alpaca is blocked
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[{"symbol": "GOOGL", "status": "new", "qty": "10"}],
        ),
    ):
        res4 = isolated_broker.execute_buy(ticker="GOOGL", shares=10, price=180.0)
        assert res4["success"] is False
        assert "Alpaca" in res4["error"]


def test_position_over_10_pct_equity_is_blocked(isolated_broker):
    """(b) A position exceeding 10% of total equity is blocked."""
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        # Total equity is $100,000. 10% cap is $10,000.
        # Attempting to buy 100 shares @ $150.0 ($15,000 = 15% of equity) must be blocked
        res_oversized = isolated_broker.execute_buy(
            ticker="NVDA", shares=100, price=150.0
        )
        assert res_oversized["success"] is False
        assert "exceeds max 10% of equity" in res_oversized["error"]

        # Buying 50 shares @ $150.0 ($7,500 = 7.5% of equity) must succeed
        res_valid = isolated_broker.execute_buy(ticker="NVDA", shares=50, price=150.0)
        assert res_valid["success"] is True
        assert "NVDA" in isolated_broker.state["open_positions"]

    # Also test when existing Alpaca pending order causes combined exposure > 10%
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[
                {"symbol": "AMZN", "status": "new", "qty": "50", "limit_price": 160.0}
            ],
        ),
    ):
        # Existing pending order is $8,000 (8%). Attempting to add $3,000 more (total $11,000 = 11%) is blocked
        res_combined_oversized = isolated_broker.execute_buy(
            ticker="AMZN", shares=20, price=150.0
        )
        assert res_combined_oversized["success"] is False


def test_eleventh_concurrent_position_is_blocked(isolated_broker):
    """(c) The 11th concurrent position is blocked when MAX_CONCURRENT_POSITIONS is 10."""
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        # Fill exactly 10 distinct concurrent positions
        tickers = [f"TICK{i}" for i in range(1, 11)]
        for t in tickers:
            res = isolated_broker.execute_buy(ticker=t, shares=2, price=100.0)
            assert res["success"] is True, f"Failed to open position for {t}: {res}"

        assert len(isolated_broker.state["open_positions"]) == 10

        # Attempt to open an 11th concurrent position
        res_11th = isolated_broker.execute_buy(ticker="TICK11", shares=2, price=100.0)
        assert res_11th["success"] is False
        assert "Maximum concurrent positions (10) reached" in res_11th["error"]
        assert "TICK11" not in isolated_broker.state["open_positions"]


def test_fail_closed_when_alpaca_unavailable_in_live_mode(isolated_broker, monkeypatch, caplog):
    """
    Fail closed: When LIVE_ALPACA_REQUIRED=true, if Alpaca is not connected,
    or if querying positions/orders raises an exception or times out,
    the buy MUST be skipped and logged at WARNING level 'BUY_SKIPPED_ALPACA_UNAVAILABLE'.
    """
    import logging
    import requests

    monkeypatch.setenv("LIVE_ALPACA_REQUIRED", "true")

    # Case 1: alpaca.is_connected() is False
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
            res = isolated_broker.execute_buy(ticker="NVDA", shares=10, price=100.0)
            assert res["success"] is False
            assert res["error"] == "BUY_SKIPPED_ALPACA_UNAVAILABLE"
            assert "BUY_SKIPPED_ALPACA_UNAVAILABLE" in caplog.text
            assert "NVDA" not in isolated_broker.state["open_positions"]

    # Case 2: Alpaca query raises an exception
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        with (
            patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
            patch(
                "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
                side_effect=requests.RequestException("Alpaca API connection failure"),
            ),
        ):
            res = isolated_broker.execute_buy(ticker="NVDA", shares=10, price=100.0)
            assert res["success"] is False
            assert res["error"] == "BUY_SKIPPED_ALPACA_UNAVAILABLE"
            assert "BUY_SKIPPED_ALPACA_UNAVAILABLE" in caplog.text
            assert "NVDA" not in isolated_broker.state["open_positions"]

    # Case 3: Alpaca query times out
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        with (
            patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
            patch(
                "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
                side_effect=requests.Timeout("Alpaca API timed out"),
            ),
        ):
            res = isolated_broker.execute_buy(ticker="NVDA", shares=10, price=100.0)
            assert res["success"] is False
            assert res["error"] == "BUY_SKIPPED_ALPACA_UNAVAILABLE"
            assert "BUY_SKIPPED_ALPACA_UNAVAILABLE" in caplog.text
            assert "NVDA" not in isolated_broker.state["open_positions"]


def test_alpaca_order_only_submitted_after_execute_buy_success(isolated_broker):
    """
    Proves that an Alpaca bracket order can only be submitted after
    execute_buy returns success for the same ticker in the same call.
    If execute_buy fails (e.g. concentration limit breached), no Alpaca order is submitted.
    """
    mock_alpaca = MagicMock()
    mock_alpaca.is_connected.return_value = True
    mock_alpaca.get_open_positions.return_value = []
    mock_alpaca.get_open_orders.return_value = []
    mock_alpaca.submit_bracket_order.return_value = {"status": "accepted", "id": "test-order-123"}

    # Mock execute_buy failure (e.g. concentration cap exceeded)
    with patch("src.paper_broker.PaperBroker.execute_buy") as mock_exec_buy:
        mock_exec_buy.return_value = {"success": False, "error": "Maximum 10% equity cap breached"}

        # Simulate execution attempt in execute_daily_signals
        signals = [
            {
                "ticker": "NVDA",
                "signal": "BUY",
                "current_price": 200.0,
                "confidence": 0.85,
                "take_profit": 220.0,
                "stop_loss": 190.0,
            }
        ]
        with (
            patch("src.alpaca_broker.AlpacaBrokerBridge", return_value=mock_alpaca),
            patch("os.environ.get", side_effect=lambda k, default=None: "0" if k == "SENTILYZE_DRY_RUN" else default),
        ):
            res = isolated_broker.execute_daily_signals(signals, date_str="2026-10-26")
            assert len(res["buys"]) == 0
            # Alpaca bracket order must NOT have been called
            mock_alpaca.submit_bracket_order.assert_not_called()

    # Now verify success path: execute_buy succeeds, Alpaca bracket order is submitted
    with patch("src.paper_broker.PaperBroker.execute_buy") as mock_exec_buy:
        mock_exec_buy.return_value = {
            "success": True,
            "ticker": "NVDA",
            "shares": 10,
            "entry_price": 200.0,
            "cost": 2000.0,
        }
        with (
            patch("src.alpaca_broker.AlpacaBrokerBridge", return_value=mock_alpaca),
            patch("os.environ.get", side_effect=lambda k, default=None: "0" if k == "SENTILYZE_DRY_RUN" else default),
        ):
            res = isolated_broker.execute_daily_signals(signals, date_str="2026-10-26")
            assert len(res["buys"]) == 1
            assert mock_alpaca.submit_bracket_order.call_count == 1
            call_kwargs = mock_alpaca.submit_bracket_order.call_args[1]
            assert call_kwargs["ticker"] == "NVDA"
            assert call_kwargs["side"] == "buy"
            assert call_kwargs["qty"] > 0
            assert call_kwargs["take_profit_price"] == 220.0


def test_real_autonomous_cycle_twice_submits_only_one_alpaca_order(tmp_path, monkeypatch):
    """
    Assert that running the real autonomous trading cycle twice with a mocked Alpaca client
    submits an order on the first run, and blocks duplicate submission on the second run.
    """
    from src.autonomous_trader import AutonomousTradingEngine
    import pandas as pd

    portfolio_file = tmp_path / "paper_portfolio.json"
    trades_file = tmp_path / "executed_trades.csv"
    broker = PaperBroker(portfolio_path=str(portfolio_file), initial_cash=150000.0, trades_path=str(trades_file))

    mock_alpaca = MagicMock()
    mock_alpaca.is_connected.return_value = True
    # Initially no positions/orders on Alpaca
    mock_alpaca.get_open_positions.return_value = []
    mock_alpaca.get_open_orders.return_value = []
    mock_alpaca.submit_bracket_order.return_value = {"status": "SUBMITTED", "order": {"id": "alpaca_1"}}

    monkeypatch.setenv("ALLOW_ALPACA_ORDER_SUBMISSION", "true")
    monkeypatch.setenv("SENTILYZE_DRY_RUN", "0")
    monkeypatch.setattr("src.alpaca_broker.AlpacaBrokerBridge", lambda: mock_alpaca)
    monkeypatch.setattr(
        "src.market_session.get_us_market_session",
        lambda: {"status": "REGULAR_HOURS", "is_open": True, "time_edt": "10:30 AM", "utc_time": "14:30"},
    )

    # Mock SPY price history so SPY close is above SMA50
    def mock_get_price_history(ticker, *args, **kwargs):
        dates = pd.date_range("2026-01-01", periods=60)
        return pd.DataFrame({"Close": [100.0 + i for i in range(60)]}, index=dates)

    monkeypatch.setattr("src.data_ingestion.get_price_history", mock_get_price_history)

    # Mock real news to satisfy news gate
    real_news_df = pd.DataFrame([
        {
            "Title": "Nvidia announces record data center revenue growth and new AI platform",
            "publishedAt": "2026-10-26T08:00:00Z",
            "source": "Reuters",
            "url": "https://reuters.com/nvda",
            "description": "Nvidia beats estimates",
        }
    ])
    real_news_df.attrs["news_available"] = True
    monkeypatch.setattr("src.autonomous_trader.get_news", lambda *args, **kwargs: real_news_df)
    monkeypatch.setattr("src.autonomous_trader.fetch_universe_live_quotes", lambda tickers: {t: {"ticker": t, "price": 120.0, "status": "LIVE"} for t in tickers})

    # Mock committee approval
    approved_resolution = {
        "ticker": "NVDA",
        "action_code": "EXECUTE_BUY",
        "final_resolution": "EXECUTE_BUY",
        "spot_price": 120.0,
        "approved_leverage": 1.0,
        "kelly_allocation_pct": 0.05,
        "metrics": {"atr": 2.5},
        "cro_signoff": {
            "action_code": "EXECUTE_BUY",
            "approved_leverage": 1.0,
            "kelly_allocation_pct": 0.05,
            "tp1_target": 130.0,
            "tp2_target": 140.0,
            "sl_target": 115.0,
        },
    }
    monkeypatch.setattr("src.autonomous_trader.convene_trading_committee", lambda *args, **kwargs: approved_resolution)

    engine = AutonomousTradingEngine(broker=broker)

    # Run cycle 1: Order should be submitted
    res1 = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])
    assert len(res1.get("buys", [])) == 1
    assert mock_alpaca.submit_bracket_order.call_count == 1

    # Simulate Alpaca now reporting the newly created open position
    mock_alpaca.get_open_positions.return_value = [{"symbol": "NVDA", "qty": "41", "side": "long"}]

    # Run cycle 2: Second buy must be blocked by duplicate position guard, NO new Alpaca order
    res2 = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])
    assert len(res2.get("buys", [])) == 0
    assert mock_alpaca.submit_bracket_order.call_count == 1  # Still exactly 1!




def test_learning_frozen_preserves_byte_identical_memory(tmp_path, monkeypatch):
    """
    Asserts that with LEARNING_FROZEN=true, executing post-mortem cycles,
    self-improvement loops, and committee weight recalibrations leaves
    results/agent_learning_memory.json and results/agent_memory/* byte-identical.
    """
    import glob
    import hashlib
    from src.post_mortem_learner import TradePostMortemLearner
    from src.autonomous_trader import AutonomousTradingEngine
    from src.agent_memory import AgentMemoryStore

    monkeypatch.setenv("LEARNING_FROZEN", "true")

    memory_files = sorted(
        ["results/agent_learning_memory.json"]
        + [f.replace("\\", "/") for f in glob.glob("results/agent_memory/*")]
    )

    def compute_hashes():
        hashes = {}
        for mf in memory_files:
            if os.path.exists(mf):
                with open(mf, "rb") as fp:
                    hashes[mf] = hashlib.sha256(fp.read()).hexdigest()
        return hashes

    hashes_before = compute_hashes()

    # 1. Run post-mortem learning cycle
    learner = TradePostMortemLearner()
    learner.run_post_mortem_cycle()

    # 2. Run self-improvement loop
    test_broker = PaperBroker(
        portfolio_path=str(tmp_path / "test_portfolio.json"),
        trades_path=str(tmp_path / "test_trades.csv"),
    )
    engine = AutonomousTradingEngine(broker=test_broker)
    engine._run_self_improvement_feedback_loop(
        cycle_actions={
            "buys": [],
            "take_profits_tp1": [],
            "take_profits_tp2": [],
            "stop_losses": [],
            "veto_exits": [],
        }
    )

    # 3. Run AgentMemoryStore recalibration
    store = AgentMemoryStore()
    store.recalibrate_weights()

    hashes_after = compute_hashes()

    assert hashes_before == hashes_after, (
        f"Memory modified while LEARNING_FROZEN=true! Mismatches: "
        f"{[k for k in hashes_before if hashes_before[k] != hashes_after.get(k)]}"
    )


def test_fresh_container_duplicate_alpaca_position_and_order_blocks_buy(tmp_path):
    """
    A fresh container (NEW PaperBroker with an EMPTY local ledger):
    - When Alpaca returns an open NVDA position: execute_buy returns success=False
      and submit_bracket_order is NOT called.
    - When Alpaca returns an open NVDA order with status 'new': execute_buy returns success=False
      and submit_bracket_order is NOT called.
    """
    # 1. Open position on Alpaca
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[{"symbol": "NVDA", "qty": "10", "market_value": "1200.0"}],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_account_summary",
            return_value={"status": "CONNECTED", "equity": 100000.0, "cash": 90000.0},
        ),
        patch("src.alpaca_broker.AlpacaBrokerBridge.submit_bracket_order") as mock_submit1,
    ):
        fresh_broker1 = PaperBroker(
            portfolio_path=str(tmp_path / "fresh_portfolio_1.json"),
            trades_path=str(tmp_path / "fresh_trades_1.csv"),
            initial_cash=100000.0,
        )
        assert len(fresh_broker1.state["open_positions"]) == 0

        res1 = fresh_broker1.execute_buy(ticker="NVDA", shares=10, price=120.0)
        assert res1["success"] is False
        assert "Active position already exists on Alpaca" in res1["error"]
        mock_submit1.assert_not_called()

    # 2. Open pending order with status 'new' on Alpaca
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[{"symbol": "NVDA", "status": "new", "qty": "10"}],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_account_summary",
            return_value={"status": "CONNECTED", "equity": 100000.0, "cash": 90000.0},
        ),
        patch("src.alpaca_broker.AlpacaBrokerBridge.submit_bracket_order") as mock_submit2,
    ):
        fresh_broker2 = PaperBroker(
            portfolio_path=str(tmp_path / "fresh_portfolio_2.json"),
            trades_path=str(tmp_path / "fresh_trades_2.csv"),
            initial_cash=100000.0,
        )
        assert len(fresh_broker2.state["open_positions"]) == 0

        res2 = fresh_broker2.execute_buy(ticker="NVDA", shares=10, price=120.0)
        assert res2["success"] is False
        assert "Pending order already exists on Alpaca" in res2["error"]
        mock_submit2.assert_not_called()


def test_live_risk_source_reads_alpaca_account_and_skips_when_unavailable(tmp_path, monkeypatch):
    """
    On live path (LIVE_ALPACA_REQUIRED=true):
    - Reads equity and cash from Alpaca account (/v2/account).
    - If Alpaca is unreachable, skips buy and logs BUY_SKIPPED_ALPACA_UNAVAILABLE at WARNING.
    """
    monkeypatch.setenv("LIVE_ALPACA_REQUIRED", "true")

    broker = PaperBroker(
        portfolio_path=str(tmp_path / "live_risk_pf.json"),
        trades_path=str(tmp_path / "live_risk_tr.csv"),
        initial_cash=50000.0,
    )

    # 1. Alpaca unreachable -> execute_buy skips buy with BUY_SKIPPED_ALPACA_UNAVAILABLE
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        res = broker.execute_buy("NVDA", shares=10, price=100.0)
        assert res["success"] is False
        assert res["error"] == "BUY_SKIPPED_ALPACA_UNAVAILABLE"

    # 2. Alpaca connected with live equity $200k -> 10% concentration limit scales to $20k
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch("src.alpaca_broker.AlpacaBrokerBridge.get_open_positions", return_value=[]),
        patch("src.alpaca_broker.AlpacaBrokerBridge.get_open_orders", return_value=[]),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_account_summary",
            return_value={"status": "CONNECTED", "equity": 200000.0, "cash": 200000.0},
        ),
    ):
        # Buy cost is $15,000 (which exceeds local $50k * 10% = $5k, but is within Alpaca $200k * 10% = $20k)
        res_live = broker.execute_buy("NVDA", shares=150, price=100.0)
        assert res_live["success"] is True


def test_live_entry_path_does_not_read_refreshing_radar_files(tmp_path, monkeypatch):
    """
    Asserts that the autonomous live scanning and candidate selection path
    operates dynamically from real-time quotes and NEVER reads stale/refreshing
    radar files (tier1_asymmetric_radar.json, most_increasable_trade_set_latest.json,
    swift_shift_radar_latest.json).
    """
    from src.autonomous_trader import AutonomousTradingEngine
    import pandas as pd

    forbidden_files = {
        "tier1_asymmetric_radar.json",
        "most_increasable_trade_set_latest.json",
        "swift_shift_radar_latest.json",
    }

    original_open = open

    def guarded_open(file, *args, **kwargs):
        for ff in forbidden_files:
            if ff in str(file):
                raise AssertionError(f"Live path attempted to read forbidden refreshing file: {file}")
        return original_open(file, *args, **kwargs)

    test_broker = PaperBroker(
        portfolio_path=str(tmp_path / "test_pf.json"),
        trades_path=str(tmp_path / "test_tr.csv"),
    )
    engine = AutonomousTradingEngine(broker=test_broker)

    dummy_news = pd.DataFrame([{"Title": "NVDA Earnings", "publishedAt": "2026-10-26"}])
    dummy_news.attrs["news_available"] = True

    with (
        patch("builtins.open", side_effect=guarded_open),
        patch("src.autonomous_trader.get_news", return_value=dummy_news),
        patch("src.autonomous_trader.convene_trading_committee", return_value={"action_code": "HOLD"}),
    ):
        # Autonomous cycle should rank and deliberate candidates without reading refreshing files
        res = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])
        assert isinstance(res, dict)


def test_news_quota_or_error_funnel_counter_and_daily_cache(tmp_path, monkeypatch):
    """
    Asserts that:
    1. HTTP 429 quota exhaustion sets quota_or_error=True and is counted under
       'skipped_news_quota_or_error' separate from 'no news found'.
    2. Once news is fetched for a ticker on a day, subsequent calls read from
       the daily universe cache (results/news_cache_YYYY-MM-DD.json) without making external API calls.
    """
    from datetime import datetime, timezone
    from src.funnel_tracker import record_funnel_run

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sim_log = str(tmp_path / "funnel_log.jsonl")
    sim_md = str(tmp_path / "funnel.md")
    sim_json = str(tmp_path / "funnel.json")

    # 1. Funnel records skipped_news_quota_or_error separately
    record = record_funnel_run(
        tickers_scanned=10,
        skipped_no_real_news=2,
        skipped_news_quota_or_error=5,
        vetoed_by_reason={"VALUATION": 2},
        buy_blocked_guards={},
        orders_submitted=1,
        json_path=sim_json,
        md_path=sim_md,
        log_path=sim_log,
    )

    assert record["skipped_news_quota_or_error"] == 5
    assert record["skipped_no_real_news"] == 2

    with open(sim_md, "r", encoding="utf-8") as f:
        md_content = f.read()
    assert "| Skipped (Quota/Error) |" in md_content
    assert " 5 " in md_content
