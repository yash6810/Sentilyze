"""
Unit tests for Congressional STOCK Act Tracker (Sprint 2, Module 2.1)
"""

import pytest
import json
from src.congress_tracker import CongressionalStockTracker, get_congressional_sentiment


@pytest.fixture
def mock_cache_file(tmp_path):
    cache = tmp_path / "mock_congress_cache.json"
    trades = [
        {
            "ticker": "NVDA",
            "representative": "Hon. John Doe",
            "party": "Democrat",
            "type": "purchase",
            "amount": "$15,001 - $50,000",
            "transaction_date": "2026-08-15",
        },
        {
            "ticker": "NVDA",
            "representative": "Hon. Jane Smith",
            "party": "Republican",
            "type": "purchase",
            "amount": "$50,001 - $100,000",
            "transaction_date": "2026-08-20",
        },
        {
            "ticker": "NVDA",
            "representative": "Hon. Bob Trader",
            "party": "Democrat",
            "type": "sale_full",
            "amount": "$1,001 - $15,000",
            "transaction_date": "2026-08-10",
        },
        {
            "ticker": "XYZ",
            "representative": "Hon. Alice Bear",
            "party": "Democrat",
            "type": "sale_full",
            "amount": "$15,001 - $50,000",
            "transaction_date": "2026-08-12",
        },
    ]
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(trades, f)
    return str(cache)


def test_analyze_ticker_with_bipartisan_cluster(mock_cache_file):
    tracker = CongressionalStockTracker(cache_file=mock_cache_file)
    res = tracker.analyze_ticker_congress_alignment("NVDA", lookback_days=180)

    assert res["status"] == "SUCCESS"
    assert res["ticker"] == "NVDA"
    assert res["purchases_count"] == 2
    assert res["sales_count"] == 1
    assert res["net_buys"] == 1
    assert res["bipartisan_cluster"] is True
    assert res["congress_conviction_score"] > 60.0
    assert "BULLISH" in res["sentiment_verdict"]


def test_analyze_ticker_bearish(mock_cache_file):
    tracker = CongressionalStockTracker(cache_file=mock_cache_file)
    res = tracker.analyze_ticker_congress_alignment("XYZ", lookback_days=180)

    assert res["status"] == "SUCCESS"
    assert res["purchases_count"] == 0
    assert res["sales_count"] == 1
    assert res["net_buys"] == -1
    assert res["congress_conviction_score"] < 40.0
    assert "BEARISH" in res["sentiment_verdict"]


def test_analyze_ticker_no_filings(mock_cache_file):
    tracker = CongressionalStockTracker(cache_file=mock_cache_file)
    res = tracker.analyze_ticker_congress_alignment("UNKNOWN_TICKER", lookback_days=180)

    assert res["status"] == "NO_RECENT_POLITICAL_TRADES"
    assert res["congress_conviction_score"] == 50.0
    assert res["total_trades_count"] == 0


def test_get_congressional_sentiment_smoke():
    res = get_congressional_sentiment("NVDA")
    assert "congress_conviction_score" in res
    assert "sentiment_verdict" in res
