"""
Unit tests for Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm.
"""

import pytest
from src.ticker_sentinel import (
    detect_peak_crest_exhaustion,
    TickerSentinel,
    TickerSentinelSwarm,
)


def test_detect_peak_crest_exhaustion_on_volume_climax():
    # Stock bought at 100.0, peaked at 110.0, now at 109.2 with 1.6x volume surge
    res = detect_peak_crest_exhaustion(
        current_price=109.2,
        entry_price=100.0,
        highest_price_seen=110.0,
        volume_ratio=1.60,
        recent_closes=[109.1, 109.2, 109.2],
    )
    assert res["exhaustion_score"] > 0.3
    assert res["peak_price"] == 110.0
    assert len(res["signals"]) > 0


def test_ticker_sentinel_lifecycle():
    sentinel = TickerSentinel(ticker="TSM", entry_price=417.52, shares=45)
    assert sentinel.ticker == "TSM"
    assert sentinel.highest_price_seen == 417.52

    # Stock rallies to 426.00
    report = sentinel.audit_tick(current_price=426.00, volume_ratio=1.35)
    assert sentinel.highest_price_seen == 426.00
    assert report["unrealized_pnl"] > 0
    assert report["status"] in ["🟢 TRACKING WAVE", "🎯 HARVEST READY"]


def test_ticker_sentinel_swarm():
    swarm = TickerSentinelSwarm()
    positions = {
        "TSM": {"entry_price": 417.52, "shares": 45},
        "ADBE": {"entry_price": 291.52, "shares": 20},
    }
    swarm.sync_open_positions(positions)
    assert len(swarm.sentinels) == 2
    assert "TSM" in swarm.sentinels
    assert "ADBE" in swarm.sentinels

    quotes = {
        "TSM": {"price": 420.00},
        "ADBE": {"price": 295.00},
    }
    reports = swarm.audit_all_sentinels(quotes)
    assert len(reports) == 2
