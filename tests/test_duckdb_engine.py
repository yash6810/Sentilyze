"""
Unit tests for DuckDB Columnar Data Lake & High-Speed Scanner Engine.
"""

import pytest
import pandas as pd
from datetime import date, timedelta

from src.duckdb_engine import DuckDBMarketEngine


@pytest.fixture
def memory_engine():
    """Provides an isolated in-memory DuckDB market engine."""
    engine = DuckDBMarketEngine(db_path=":memory:")
    yield engine
    engine.close()


@pytest.fixture
def sample_bars_df():
    """Generates synthetic daily bars for testing."""
    dates = [date(2026, 9, 1) + timedelta(days=i) for i in range(10)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [98.0 + i for i in range(10)],
            "close": [102.0 + i for i in range(10)],
            "volume": [1000000 + i * 50000 for i in range(10)],
            "rsi": [45.0 + i * 2.0 for i in range(10)],
            "sma50": [95.0 + i for i in range(10)],
            "sma200": [90.0 for _ in range(10)],
            "atr": [2.5 for _ in range(10)],
            "vwap": [101.0 + i for i in range(10)],
        }
    )


def test_duckdb_schema_initialization(memory_engine):
    """Verify tables are created in the database."""
    tables_df = memory_engine.query("SHOW TABLES;")
    tables = tables_df["name"].tolist() if not tables_df.empty else []
    assert "daily_bars" in tables
    assert "universe_registry" in tables
    assert "sec_catalysts" in tables


def test_duckdb_ingest_and_query(memory_engine, sample_bars_df):
    """Verify bar ingestion and fast SQL querying."""
    rows = memory_engine.ingest_bars(sample_bars_df, "NVDA")
    assert rows == 10

    # Query back with deterministic ordering
    res = memory_engine.query(
        "SELECT * FROM daily_bars WHERE ticker = 'NVDA' ORDER BY date ASC;"
    )
    assert len(res) == 10
    assert "close" in res.columns
    assert float(res["close"].iloc[-1]) == 111.0


def test_duckdb_momentum_scanner(memory_engine, sample_bars_df):
    """Verify vectorized momentum breakout scanner."""
    memory_engine.ingest_bars(sample_bars_df, "NVDA")

    # Lower RSI ticker
    low_df = sample_bars_df.copy()
    low_df["rsi"] = 35.0
    memory_engine.ingest_bars(low_df, "INTC")

    # Screen between 50 and 70 RSI
    screened = memory_engine.scan_momentum_breakouts(min_rsi=50.0, max_rsi=70.0)
    assert not screened.empty
    assert "NVDA" in screened["ticker"].values
    assert "INTC" not in screened["ticker"].values


def test_duckdb_golden_cross_scanner(memory_engine, sample_bars_df):
    """Verify golden cross scanner detection."""
    # Bullish cross: SMA50 (104) >= SMA200 (90)
    memory_engine.ingest_bars(sample_bars_df, "NVDA")

    crosses = memory_engine.scan_golden_crosses()
    assert not crosses.empty
    assert "NVDA" in crosses["ticker"].values
    assert crosses["spread_pct"].iloc[0] > 0.0


def test_duckdb_coverage_summary(memory_engine, sample_bars_df):
    """Verify coverage summary reporting."""
    memory_engine.ingest_bars(sample_bars_df, "AAPL")
    memory_engine.ingest_bars(sample_bars_df, "MSFT")

    summary = memory_engine.get_coverage_summary()
    assert summary["total_tickers"] == 2
    assert summary["total_bars"] == 20
    assert summary["earliest_date"] is not None
    assert summary["latest_date"] is not None
