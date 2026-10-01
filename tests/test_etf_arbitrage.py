"""
Unit tests for ETF Intraday NAV Arbitrage Vector (Sprint 2, Module 2.9)
"""

import pytest
from src.etf_arbitrage import (
    calculate_inav,
    calculate_etf_nav_spread,
    evaluate_etf_arbitrage_vector,
)


def test_calculate_inav():
    weights = {"NVDA": 0.5, "AAPL": 0.5}
    prices = {"NVDA": 120.0, "AAPL": 180.0}
    # (0.5 * 120 + 0.5 * 180) = 150.0
    inav = calculate_inav(weights, prices, target_scale=1.0)
    assert inav == 150.0


def test_calculate_etf_nav_spread_premium():
    # ETF at $202, iNAV at $200 (+1% premium)
    res = calculate_etf_nav_spread(etf_price=202.0, inav=200.0)
    assert res["spread_pct"] == 1.0
    assert res["spread_bps"] == 100.0
    assert res["arbitrage_regime"] == "ETF_PREMIUM_AP_CREATION_FLOW"
    assert res["constituent_flow_prediction"] == "INSTITUTIONAL_BUY_TAILWIND"


def test_calculate_etf_nav_spread_discount():
    # ETF at $198, iNAV at $200 (-1% discount)
    res = calculate_etf_nav_spread(etf_price=198.0, inav=200.0)
    assert res["spread_pct"] == -1.0
    assert res["spread_bps"] == -100.0
    assert res["arbitrage_regime"] == "ETF_DISCOUNT_AP_REDEMPTION_DRAG"
    assert res["constituent_flow_prediction"] == "INSTITUTIONAL_SELL_HEADWIND"


def test_evaluate_etf_arbitrage_vector_smoke():
    custom_prices = {"NVDA": 130.0, "TSM": 170.0, "AVGO": 160.0}
    res = evaluate_etf_arbitrage_vector("SMH", custom_constituent_prices=custom_prices)
    assert res["status"] == "SUCCESS"
    assert res["etf_ticker"] == "SMH"
    assert "spread_pct" in res
    assert "constituent_flow_prediction" in res
