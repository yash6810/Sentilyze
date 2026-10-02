"""
Unit tests for Autonomous Benzinga Breaking News Watchdog (Option 1).
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.benzinga_watchdog import BenzingaNewsWatchdog, get_benzinga_watchdog
from src.paper_broker import PaperBroker


@pytest.fixture
def mock_temp_broker(tmp_path):
    port_file = str(tmp_path / "test_portfolio.json")
    trades_file = str(tmp_path / "test_trades.csv")
    broker = PaperBroker(
        initial_cash=100000.0,
        portfolio_file=port_file,
        trades_file=trades_file,
    )
    return broker


def test_watchdog_hash_and_deduplication(tmp_path):
    s_file = str(tmp_path / "watchdog_state_hash.json")
    watchdog = BenzingaNewsWatchdog(webhook_url=None, state_file=s_file)
    h1 = watchdog.hash_headline("Nvidia Secures $10B AI Chip Order")
    h2 = watchdog.hash_headline("   nvidia secures $10b ai chip order   ")
    assert h1 == h2
    assert len(h1) == 16


def test_watchdog_poll_and_dispatch(mock_temp_broker, monkeypatch, tmp_path):
    mock_articles = [
        {
            "title": "Amazon Announces $1B Cloud Expansion Deal",
            "published": "Fri, 02 Oct 2026 12:00:00 GMT",
            "extracted_tickers": ["AMZN"],
            "sentiment_score": 0.8,
            "sentiment_label": "BULLISH",
            "link": "https://example.com/amzn",
        }
    ]

    mock_scan_rep = {
        "status": "COMPLETED",
        "top_evaluated_stocks": [
            {
                "ticker": "AMZN",
                "current_price": 220.0,
                "benzinga_sentiment": "BULLISH",
                "net_catalyst_score": 0.8,
                "committee_verdict": "STRONG_BUY",
                "committee_conviction_pct": 82.5,
                "trade_action": "BUY_SUBMITTED",
                "latest_headline": "Amazon Announces $1B Cloud Expansion Deal",
            }
        ],
        "trade_actions_executed": ["BUY 10 AMZN @ $220.00"],
    }

    monkeypatch.setattr(
        "src.benzinga_watchdog.fetch_top_benzinga_news_wire",
        lambda limit=60: mock_articles,
    )
    monkeypatch.setattr(
        "src.benzinga_watchdog.scan_and_trade_benzinga_catalysts",
        lambda execute_paper, max_stocks_to_evaluate, broker: mock_scan_rep,
    )

    s_file = str(tmp_path / "watchdog_state_poll.json")
    watchdog = BenzingaNewsWatchdog(
        webhook_url="https://discord.com/api/webhooks/mock",
        state_file=s_file,
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=204)
        result = watchdog.poll_and_dispatch(
            auto_trade=True,
            min_catalyst_score=0.4,
            broker=mock_temp_broker,
        )

        assert result["status"] == "COMPLETED"
        assert result["articles_scanned"] == 1
        assert result["new_catalysts_detected"] == 1
        assert "AMZN" in result["alerts_dispatched"]
        assert len(result["trades_executed"]) == 1
        assert watchdog.total_alerts_dispatched == 1

        # Second poll with same headline should deduplicate (0 new catalysts)
        res2 = watchdog.poll_and_dispatch(
            auto_trade=True,
            min_catalyst_score=0.4,
            broker=mock_temp_broker,
        )
        assert res2["new_catalysts_detected"] == 0


def test_watchdog_background_thread_start_stop(tmp_path):
    s_file = str(tmp_path / "watchdog_state_thread.json")
    watchdog = BenzingaNewsWatchdog(webhook_url=None, state_file=s_file)
    watchdog.start_background_watchdog(interval_seconds=1, auto_trade=False)
    assert watchdog._daemon_thread is not None
    assert watchdog._daemon_thread.is_alive()
    watchdog.stop_background_watchdog()
    assert not watchdog._daemon_thread.is_alive()
