from unittest.mock import patch, MagicMock
from src.reddit_premarket_station import (
    scrape_station_ticker_sentiment,
    fetch_4station_premarket_intelligence,
    _fetch_subreddit_rss_entries,
)


def test_scrape_station_ticker_sentiment_fallback():
    res = scrape_station_ticker_sentiment("NVDA", station_id="wsb")
    assert res["station_id"] == "wsb"
    assert "bullish_pct" in res
    assert "normalized_score" in res
    assert len(res["threads"]) > 0


def test_fetch_4station_premarket_intelligence():
    intel = fetch_4station_premarket_intelligence("NVDA")
    assert intel["ticker"] == "NVDA"
    assert len(intel["stations"]) == 4
    assert -1.0 <= intel["composite_score"] <= 1.0
    assert 0.0 <= intel["composite_conviction_pct"] <= 100.0
    assert "regime" in intel
    assert "color" in intel


@patch("requests.Session.get")
def test_fetch_subreddit_rss_entries_mock(mock_get):
    xml_sample = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <title>What Are Your Moves Tomorrow, August 28: $NVDA Earnings Squeeze</title>
            <link href="https://reddit.com/r/wallstreetbets/test"/>
            <updated>2026-08-28T00:00:00+00:00</updated>
        </entry>
    </feed>"""
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.content = xml_sample
    mock_get.return_value = mock_res

    entries = _fetch_subreddit_rss_entries("wallstreetbets", limit=2)
    assert len(entries) == 1
    assert "NVDA" in entries[0]["title"]
