"""
Unit tests for src/benzinga_news_trader.py.
Verifies Benzinga news scraping, ticker extraction, catalyst scoring,
and paper trade routing with mock isolation.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock

from src.benzinga_news_trader import (
    extract_tickers_from_headline,
    fetch_top_benzinga_news_wire,
    scan_and_trade_benzinga_catalysts,
    get_latest_benzinga_scan_report,
)
from src.paper_broker import PaperBroker


def test_extract_tickers_from_headline():
    h1 = "Broadcom Bets $42 Billion on Anthropic - Broadcom (NASDAQ:AVGO) - Benzinga"
    assert extract_tickers_from_headline(h1) == ["AVGO"]

    h2 = "Full Transcript: Nike Q1 2027 Earnings Call - Nike (NYSE:NKE) - Benzinga"
    assert extract_tickers_from_headline(h2) == ["NKE"]

    h3 = "Micron (MU) Delivers Q4 Double Beat, Sees 'Even Stronger' Year Ahead"
    assert extract_tickers_from_headline(h3) == ["MU"]

    h4 = "NIO Stock Hits 52-Week Low: What's Happening?"
    assert extract_tickers_from_headline(h4) == ["NIO"]

    # Non-ticker words should be excluded
    h5 = "Best AI Tech Stocks Under $10 Right Now - Benzinga"
    assert extract_tickers_from_headline(h5) == []


def test_fetch_top_benzinga_news_wire_mocked():
    sample_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <item>
                <title>Micron's $150B Contract Book Tops Revenue - Micron (NASDAQ:MU) - Benzinga</title>
                <pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate>
                <link>https://www.benzinga.com/news/123</link>
            </item>
            <item>
                <title>Meta Faces Up To $40 Billion Penalty In Lawsuit - Benzinga</title>
                <pubDate>Fri, 02 Oct 2026 08:00:00 GMT</pubDate>
                <link>https://www.benzinga.com/news/456</link>
            </item>
        </channel>
    </rss>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = sample_xml

    with patch("requests.get", return_value=mock_resp):
        items = fetch_top_benzinga_news_wire(limit=10)
        assert len(items) == 2
        assert items[0]["extracted_tickers"] == ["MU"]
        assert items[0]["catalyst_sentiment"] == "BULLISH"
        assert items[1]["catalyst_sentiment"] == "BEARISH"


def test_scan_and_trade_benzinga_catalysts_mocked(tmp_path):
    p_file = str(tmp_path / "mock_portfolio.json")
    t_file = str(tmp_path / "mock_trades.csv")
    broker = PaperBroker(portfolio_path=p_file, trades_path=t_file)
    broker.state = {
        "cash": 100000.0,
        "total_equity": 100000.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "equity_history": [],
        "open_positions": {},
    }

    mock_articles = [
        {
            "title": "Akamai Stock Soars on $11.6B Anthropic Deal - Akamai (NASDAQ:AKAM)",
            "published_at": "Fri, 02 Oct 2026 10:00:00 GMT",
            "url": "https://benzinga.com/1",
            "extracted_tickers": ["AKAM"],
            "catalyst_sentiment": "BULLISH",
            "sentiment_score": 0.80,
        }
    ]

    with patch(
        "src.benzinga_news_trader.fetch_top_benzinga_news_wire",
        return_value=mock_articles,
    ):
        with patch(
            "src.benzinga_news_trader.fetch_live_quote",
            return_value={"price": 105.0, "change_pct": 3.2},
        ):
            with patch(
                "src.benzinga_news_trader.convene_trading_committee",
                return_value={"verdict": "BUY", "conviction": 75.0, "votes": {}},
            ):
                test_scan_file = str(tmp_path / "benzinga_test.json")
                with patch(
                    "src.benzinga_news_trader.BENZINGA_SCAN_FILE", test_scan_file
                ):
                    rep = scan_and_trade_benzinga_catalysts(
                        execute_paper=True,
                        max_stocks_to_evaluate=2,
                        broker=broker,
                    )
                    assert rep["total_articles_scraped"] == 1
                    assert rep["total_tickers_identified"] == 1
                    assert len(rep["top_evaluated_stocks"]) == 1
                    assert rep["top_evaluated_stocks"][0]["ticker"] == "AKAM"
                    assert rep["top_evaluated_stocks"][0]["trade_action"] == "BUY"
                    assert os.path.exists(test_scan_file)
