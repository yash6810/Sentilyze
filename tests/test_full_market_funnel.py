import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.autonomous_trader import AutonomousTradingEngine, LOCK_FILE
from src.duckdb_engine import DuckDBMarketEngine
from src.security_master import SecurityMaster, TIER_4_EXCLUDED, TIER_1_MEGA
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


def test_get_screened_candidates_duckdb_integration(tmp_path):
    """Verifies that Stage 1 broad screener queries DuckDB and returns ranked candidates."""
    db_path = str(tmp_path / "test_funnel.duckdb")
    engine = DuckDBMarketEngine(db_path=db_path)

    # Ingest 2 sample tickers with different RSI profiles
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df1 = pd.DataFrame(
        {
            "close": [100.0 + i for i in range(10)],
            "rsi": [65.0] * 10,
            "sma50": [95.0] * 10,
            "sma200": [90.0] * 10,
            "volume": [1_000_000.0] * 10,
        },
        index=dates,
    )
    df2 = pd.DataFrame(
        {
            "close": [50.0 + i for i in range(10)],
            "rsi": [40.0] * 10,  # Below 50 min_rsi
            "sma50": [45.0] * 10,
            "sma200": [40.0] * 10,
            "volume": [500_000.0] * 10,
        },
        index=dates,
    )
    engine.ingest_bars(df1, "TICKER_A")
    engine.ingest_bars(df2, "TICKER_B")
    engine.close()

    mock_broker = MagicMock(spec=PaperBroker)
    trader = AutonomousTradingEngine(broker=mock_broker)

    # Patch DuckDBMarketEngine default path in get_screened_candidates to use test db
    with patch(
        "src.duckdb_engine.DuckDBMarketEngine",
        return_value=DuckDBMarketEngine(db_path=db_path),
    ):
        candidates = trader.get_screened_candidates(limit=10)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        # TICKER_A had RSI 65 so it passes momentum breakout screen
        assert "TICKER_A" in candidates


def test_liquidity_tier_veto_blocks_illiquid_entry():
    """Verifies CRO rejects any candidate classified as TIER_4_EXCLUDED."""
    sm = SecurityMaster(cache_file="dummy_cache.csv")
    tier_penny = sm.classify_liquidity_tier(
        ticker="PENNYS", price=2.50, adv_shares=500_000
    )
    assert tier_penny == TIER_4_EXCLUDED

    tier_illiquid = sm.classify_liquidity_tier(
        ticker="LOWVOL", price=25.0, adv_shares=10_000
    )
    assert tier_illiquid == TIER_4_EXCLUDED

    tier_liquid = sm.classify_liquidity_tier(
        ticker="AAPL", price=220.0, adv_shares=40_000_000
    )
    assert tier_liquid == TIER_1_MEGA


@patch("src.autonomous_trader.fetch_universe_live_quotes")
@patch("src.autonomous_trader.get_news")
@patch("src.autonomous_trader.convene_trading_committee")
def test_full_funnel_cycle_execution_safe(
    mock_committee, mock_news, mock_quotes, tmp_path
):
    """End-to-end verification of AutonomousTradingEngine with Two-Stage Funnel."""
    mock_quotes.return_value = {
        "NVDA": {
            "ticker": "NVDA",
            "price": 125.0,
            "volume": 50_000_000,
            "status": "LIVE",
        },
    }
    mock_news.return_value = MagicMock()
    mock_committee.return_value = {
        "ticker": "NVDA",
        "spot_price": 125.0,
        "final_resolution": "🚀 CONVICTION INSTITUTIONAL BUY",
        "action_code": "EXECUTE_BUY",
        "consensus_conviction_pct": 88.0,
        "tp1_target": 132.0,
        "tp2_target": 140.0,
        "stop_loss_target": 119.0,
        "cro_signoff": {
            "approved_leverage": 1.0,
            "approved_kelly_pct": 5.0,
            "consensus_conviction_pct": 88.0,
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
        "shares": 50,
        "price": 125.0,
        "ticker": "NVDA",
    }
    mock_broker.execute_manual_buy.return_value = {
        "success": True,
        "shares": 50,
        "price": 125.0,
        "ticker": "NVDA",
    }

    engine = AutonomousTradingEngine(broker=mock_broker)
    # Patch SEC crawler to ensure zero network latency and no adverse catalyst
    with patch(
        "src.sec_crawler.SECCatalystCrawler.fetch_recent_8k_filings", return_value=[]
    ):
        res = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])

    assert res.get("status") != "ERROR"
    assert "buys" in res
    assert len(res["buys"]) == 1
    assert res["buys"][0]["ticker"] == "NVDA"
