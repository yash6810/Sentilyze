import pytest
from src.insider_signals import (
    fetch_insider_transactions,
    calculate_insider_conviction_score,
    scan_universe_insider_catalysts,
)


def test_fetch_insider_transactions():
    txs = fetch_insider_transactions("NVDA", days_back=60)
    assert isinstance(txs, list)
    if txs:
        assert "ticker" in txs[0]
        assert "officer_name" in txs[0]
        assert "value_usd" in txs[0]


def test_calculate_insider_conviction_score():
    score_data = calculate_insider_conviction_score("IEX", days_back=90)
    assert "conviction_score" in score_data
    assert 0.0 <= score_data["conviction_score"] <= 100.0
    assert "signal" in score_data
    assert isinstance(score_data["cluster_buy_detected"], bool)


def test_scan_universe_insider_catalysts():
    tickers = ["NVDA", "AAPL", "IEX", "DE"]
    ranked = scan_universe_insider_catalysts(tickers, top_n=3)
    assert len(ranked) <= 3
    if len(ranked) > 1:
        assert ranked[0]["conviction_score"] >= ranked[1]["conviction_score"]
