"""
Unit Tests for SEC EDGAR Form 4 Insider Cluster Buying Radar.
Validates CIK resolution, cluster detection algorithm, and conviction index generation.
"""

import pytest
from src.sec_form4_crawler import (
    SECForm4InsiderCrawler,
    get_insider_cluster_radar,
    TICKER_CIK_MAP,
)


def test_get_cik_resolution():
    crawler = SECForm4InsiderCrawler()
    nvda_cik = crawler.get_cik("NVDA")
    assert nvda_cik == "0001045810"
    assert len(nvda_cik) == 10

    unknown_cik = crawler.get_cik("UNKNOWN_CO")
    assert unknown_cik == "0000000000"


def test_mock_form4_filings():
    crawler = SECForm4InsiderCrawler()
    filings = crawler.fetch_recent_form4_filings("TEST")
    assert len(filings) == 2
    assert filings[0]["ticker"] == "TEST"
    assert "form4.xml" in filings[0]["document"]


def test_cluster_detection_logic(monkeypatch):
    crawler = SECForm4InsiderCrawler()
    # Mock filings: 3 filings within 3 days (Cluster of 3)
    mock_filings = [
        {"ticker": "NVDA", "filing_date": "2026-09-20"},
        {"ticker": "NVDA", "filing_date": "2026-09-21"},
        {"ticker": "NVDA", "filing_date": "2026-09-22"},
    ]
    monkeypatch.setattr(
        crawler,
        "fetch_recent_form4_filings",
        lambda ticker, days_lookback=30: mock_filings,
    )

    report = crawler.analyze_insider_cluster_signals("NVDA")
    assert report["cluster_buy_detected"] is True
    assert report["cluster_size"] == 3
    assert report["insider_conviction_score"] >= 80.0
    assert "COHEN-MALLOY CLUSTER DETECTED" in report["discretionary_signal"]


def test_single_filing_routine_activity(monkeypatch):
    crawler = SECForm4InsiderCrawler()
    # Single filing: Routine insider activity, no cluster
    mock_filings = [{"ticker": "AAPL", "filing_date": "2026-09-15"}]
    monkeypatch.setattr(
        crawler,
        "fetch_recent_form4_filings",
        lambda ticker, days_lookback=30: mock_filings,
    )

    report = crawler.analyze_insider_cluster_signals("AAPL")
    assert report["cluster_buy_detected"] is False
    assert report["cluster_size"] == 0
    assert report["insider_conviction_score"] == 55.0


def test_get_insider_cluster_radar_wrapper(monkeypatch):
    crawler_instance = SECForm4InsiderCrawler()
    monkeypatch.setattr(
        crawler_instance,
        "fetch_recent_form4_filings",
        lambda ticker, days_lookback=30: crawler_instance._generate_mock_form4_data(
            ticker
        ),
    )
    res = crawler_instance.analyze_insider_cluster_signals("MOCK")
    assert isinstance(res, dict)
    assert "insider_conviction_score" in res
    assert "cluster_buy_detected" in res
