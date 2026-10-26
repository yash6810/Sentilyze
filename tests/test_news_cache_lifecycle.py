"""
Tests for Daily News Cache Lifecycle (Zero-Network):
1. If the first fetch for a ticker returns empty due to quota_or_error, the daily cache
   is NOT written for that ticker, and the subsequent run retries live fetch.
2. A ticker missing from the daily cache is fetched live and written to the cache.
3. The daily cache file pattern (results/news_cache_*.json) is committed by the live workflow.
"""

import os
import json
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


def test_news_cache_quota_retry_and_live_fetch(tmp_path, monkeypatch):
    import src.data_ingestion as di

    # Use a simulated date so we isolate the cache file
    fake_now = datetime(2030, 5, 20, 14, 0, 0, tzinfo=timezone.utc)
    target_cache_file = os.path.join("results", "news_cache_2030-05-20.json")

    # Clean up before and after
    if os.path.exists(target_cache_file):
        os.remove(target_cache_file)

    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return fake_now

    monkeypatch.setattr(di, "datetime", MockDatetime)
    monkeypatch.setattr(di, "DATA_DIR", str(tmp_path / "data"))

    try:
        # Part (a): First fetch returns empty with quota hit
        quota_hit_df = pd.DataFrame()
        quota_hit_df.attrs["quota_hit"] = True

        empty_df = pd.DataFrame()

        live_fetch_count = 0

        def mock_finnhub(ticker, *args, **kwargs):
            nonlocal live_fetch_count
            live_fetch_count += 1
            return quota_hit_df

        with (
            patch.object(di, "_fetch_google_news_rss", return_value=empty_df),
            patch.object(di, "_fetch_yfinance_news", return_value=empty_df),
            patch.object(di, "_fetch_finnhub_news", side_effect=mock_finnhub),
            patch.object(di, "_fetch_marketaux_news", return_value=empty_df),
            patch.object(di, "_fetch_polygon_news_feed", return_value=empty_df),
            patch.object(di, "_fetch_sec_8k_news", return_value=empty_df),
            patch.object(di, "_fetch_benzinga_news", return_value=empty_df),
            patch.object(di, "_fetch_reddit_financial_news", return_value=empty_df),
            patch("src.universal_web_scraper.scrape_ticker_news_from_web", return_value=empty_df),
            patch.dict(os.environ, {"NEWS_API_KEY": ""}, clear=False),
        ):
            res_df = di.get_news("NVDA", use_cache=True)
            assert res_df.empty
            assert res_df.attrs.get("quota_or_error") is True
            assert live_fetch_count == 1

            # Assert cache file either does not exist or does NOT contain NVDA
            if os.path.exists(target_cache_file):
                with open(target_cache_file, "r", encoding="utf-8") as f:
                    cache_content = json.load(f)
                assert "NVDA" not in cache_content, "Quota-exhausted ticker must NOT be cached!"

            # Next run retries live fetch because NVDA was not cached
            success_df = pd.DataFrame([
                {
                    "Title": "NVIDIA Breakthrough AI Chip Unveiled",
                    "publishedAt": "2030-05-20T12:00:00Z",
                    "source": "Mock Wire",
                    "url": "https://example.com/nvda",
                    "description": "NVIDIA announcements today."
                }
            ])
            with patch.object(di, "_fetch_google_news_rss", return_value=success_df):
                retry_df = di.get_news("NVDA", use_cache=True)
                assert not retry_df.empty
                assert len(retry_df) == 1
                assert retry_df.attrs.get("news_available") is True
                assert retry_df.attrs.get("quota_or_error") is False
                assert "NVIDIA Breakthrough AI Chip Unveiled" in retry_df["Title"].values

            # Part (b): Ticker missing from cache is fetched live, cached, and subsequent read uses cache
            assert os.path.exists(target_cache_file)
            with open(target_cache_file, "r", encoding="utf-8") as f:
                updated_cache = json.load(f)
            assert "NVDA" in updated_cache
            assert len(updated_cache["NVDA"]) == 1

            # Fetching again should hit the daily cache without calling any live fetch functions
            with patch.object(di, "_fetch_google_news_rss", side_effect=AssertionError("Should hit daily cache!")):
                cached_res = di.get_news("NVDA", use_cache=True)
                assert not cached_res.empty
                assert len(cached_res) == 1
                assert cached_res.attrs.get("news_available") is True

    finally:
        if os.path.exists(target_cache_file):
            try:
                os.remove(target_cache_file)
            except Exception:
                pass


def test_daily_cache_file_is_committed_by_live_workflow():
    """
    Part (c): Verify that the live autonomous trader workflow commits daily news cache files.
    """
    workflow_path = os.path.join(".github", "workflows", "autonomous_live_trader.yml")
    assert os.path.exists(workflow_path), "Workflow file must exist"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow_content = f.read()

    # Must stage results/news_cache_*.json
    assert "results/news_cache_*.json" in workflow_content, (
        "Live trader workflow must stage results/news_cache_*.json"
    )

    # Must commit state
    assert "git commit" in workflow_content, (
        "Live trader workflow must commit staged state"
    )
