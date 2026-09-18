import os
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.security_master import (
    SecurityMaster,
    SECTOR_TECH,
    SECTOR_HEALTH,
    SECTOR_ENERGY,
    SECTOR_ETF,
    SECTOR_OTHER,
)
from src.duckdb_engine import DuckDBMarketEngine
from src.autonomous_trader import AutonomousTradingEngine, LOCK_FILE
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


def test_sector_classification():
    sm = SecurityMaster(cache_file="dummy_sec.csv")

    assert sm.get_ticker_sector("NVDA") == SECTOR_TECH
    assert sm.get_ticker_sector("AAPL") == SECTOR_TECH
    assert sm.get_ticker_sector("MSFT") == SECTOR_TECH
    assert sm.get_ticker_sector("LLY") == SECTOR_HEALTH
    assert sm.get_ticker_sector("XOM") == SECTOR_ENERGY
    assert sm.get_ticker_sector("SPY") == SECTOR_ETF
    assert sm.get_ticker_sector("UNKNOWN_SYM_123") == SECTOR_OTHER


def test_duckdb_regime_screening(tmp_path):
    db_path = str(tmp_path / "test_regime.duckdb")
    engine = DuckDBMarketEngine(db_path=db_path)

    dates = pd.date_range("2026-01-01", periods=10, freq="D")

    # High momentum asset (RSI 65)
    df_bull = pd.DataFrame(
        {
            "close": [100.0] * 10,
            "rsi": [65.0] * 10,
            "sma50": [95.0] * 10,
            "sma200": [90.0] * 10,
            "volume": [1_000_000.0] * 10,
        },
        index=dates,
    )
    # Oversold dip asset (RSI 35)
    df_crisis = pd.DataFrame(
        {
            "close": [50.0] * 10,
            "rsi": [35.0] * 10,
            "sma50": [55.0] * 10,
            "sma200": [60.0] * 10,
            "volume": [800_000.0] * 10,
        },
        index=dates,
    )

    engine.ingest_bars(df_bull, "BULL_STOCK")
    engine.ingest_bars(df_crisis, "DIP_STOCK")

    # Bull regime: picks BULL_STOCK (momentum)
    bull_cands = engine.get_full_market_candidates(limit=5, regime="BULL_EXPANSION")
    assert "BULL_STOCK" in bull_cands

    # Crisis regime: picks DIP_STOCK (oversold mean reversion)
    crisis_cands = engine.get_full_market_candidates(limit=5, regime="HIGH_VOL_CRISIS")
    assert "DIP_STOCK" in crisis_cands

    engine.close()


@patch("src.autonomous_trader.fetch_universe_live_quotes")
@patch("src.autonomous_trader.get_news")
@patch("src.autonomous_trader.convene_trading_committee")
def test_sector_quota_shield_veto(mock_committee, mock_news, mock_quotes):
    """Verifies that the Sector Quota Shield vetoes adding a 3rd stock to an already saturated sector."""
    # Portfolio already holds 2 Tech stocks: NVDA and AAPL
    mock_broker = MagicMock(spec=PaperBroker)
    mock_broker.state = {
        "cash": 100000.0,
        "open_positions": {
            "NVDA": {"shares": 10, "entry_price": 120.0, "current_price": 125.0},
            "AAPL": {"shares": 15, "entry_price": 200.0, "current_price": 210.0},
        },
        "closed_trades": [],
        "total_trades": 2,
        "winning_trades": 2,
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

    # Deliberation gives strong BUY for MSFT (3rd Tech stock)
    mock_quotes.return_value = {
        "MSFT": {
            "ticker": "MSFT",
            "price": 400.0,
            "volume": 20_000_000,
            "status": "LIVE",
        },
    }
    mock_news.return_value = MagicMock()
    mock_committee.return_value = {
        "ticker": "MSFT",
        "spot_price": 400.0,
        "final_resolution": "🚀 CONVICTION INSTITUTIONAL BUY",
        "action_code": "EXECUTE_BUY",
        "consensus_conviction_pct": 90.0,
        "tp1_target": 420.0,
        "tp2_target": 440.0,
        "stop_loss_target": 385.0,
        "cro_signoff": {"approved_leverage": 1.0, "approved_kelly_pct": 8.0},
    }

    engine = AutonomousTradingEngine(broker=mock_broker)

    # Patch correlation shield to allow, but sector shield must veto
    with patch(
        "src.correlation_shield.check_correlation_shield",
        return_value={"allowed": True},
    ):
        with patch(
            "src.sec_crawler.SECCatalystCrawler.fetch_recent_8k_filings",
            return_value=[],
        ):
            res = engine.run_autonomous_cycle(candidate_tickers=["MSFT"])

    # MSFT must be rejected because Tech sector is already at 2/2 capacity
    assert len(res["buys"]) == 0
