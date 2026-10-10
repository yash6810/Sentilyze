"""
Unit tests for Phase 3 24/7 Multi-Market Gateway, Meta-Labeling Filter, and Corwin-Schultz Spread.
STRICT PORTFOLIO PRESERVATION: Uses tmp_path and mock broker, zero impact on results/ files.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.market_session import is_crypto_asset, is_asset_market_open
from src.meta_labeling_filter import evaluate_meta_labeling_barrier_probability
from src.paper_broker import PaperBroker, compute_corwin_schultz_spread


def test_crypto_asset_detection():
    assert is_crypto_asset("BTC-USD") is True
    assert is_crypto_asset("ETH-USD") is True
    assert is_crypto_asset("SOL-USD") is True
    assert is_crypto_asset("BTC") is True
    assert is_crypto_asset("AAPL") is False
    assert is_crypto_asset("NVDA") is False
    assert is_crypto_asset("SPY") is False


def test_asset_market_open_crypto_24_7():
    # Crypto should ALWAYS be open 24/7 regardless of day or hour
    assert is_asset_market_open("BTC-USD") is True
    assert is_asset_market_open("ETH-USD") is True


def test_meta_labeling_barrier_probability():
    # 1. High-Conviction Bullish Setup (Trend + Good RSI + Positive OBI)
    res_bullish = evaluate_meta_labeling_barrier_probability(
        ticker="NVDA",
        spot_price=200.0,
        primary_conviction_pct=85.0,
        atr_14=5.0,
        rsi_val=55.0,
        obi_val=0.60,
        arar_score=350.0,
        is_above_sma200=True,
        vix_level=16.0,
    )
    assert res_bullish["approved"] is True
    assert res_bullish["barrier_touch_prob"] > 0.60
    assert res_bullish["meta_label"] == 1

    # 2. Downtrend Asset (Below SMA200) -> Should be vetoed by Meta-Labeler
    res_downtrend = evaluate_meta_labeling_barrier_probability(
        ticker="WEAK",
        spot_price=50.0,
        primary_conviction_pct=60.0,
        atr_14=2.0,
        rsi_val=45.0,
        obi_val=0.0,
        arar_score=0.0,
        is_above_sma200=False,
        vix_level=18.0,
    )
    assert res_downtrend["approved"] is False
    assert res_downtrend["meta_label"] == 0

    # 3. Severe Overbought Asset (RSI > 75) -> Should be vetoed
    res_overbought = evaluate_meta_labeling_barrier_probability(
        ticker="EXP",
        spot_price=100.0,
        primary_conviction_pct=75.0,
        atr_14=3.0,
        rsi_val=78.0,
        obi_val=0.20,
        is_above_sma200=True,
    )
    assert res_overbought["approved"] is False


def test_corwin_schultz_spread_computation():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "High": [
                102.0,
                103.5,
                105.0,
                104.2,
                106.0,
                105.5,
                107.0,
                108.0,
                107.5,
                109.0,
            ],
            "Low": [98.0, 99.0, 101.0, 100.5, 102.0, 101.0, 103.0, 104.0, 103.5, 105.0],
            "Close": [
                100.0,
                102.0,
                103.0,
                102.5,
                104.0,
                103.5,
                105.0,
                106.0,
                105.5,
                107.0,
            ],
        },
        index=dates,
    )
    spread = compute_corwin_schultz_spread(df)
    assert 0.0001 <= spread <= 0.05


def test_fractional_crypto_paper_broker_execution(tmp_path):
    p_file = tmp_path / "mock_paper_portfolio.json"
    p_file.write_text(
        json.dumps(
            {
                "total_equity": 100000.0,
                "cash": 100000.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "open_positions": {},
                "closed_trades": [],
                "winning_trades": 0,
                "losing_trades": 0,
                "total_trades": 0,
                "win_rate": 0.0,
                "equity_history": [],
            }
        ),
        encoding="utf-8",
    )

    broker = PaperBroker(portfolio_path=str(p_file), initial_cash=100000.0)

    # Execute a fractional crypto order (BTC at $65,000)
    signals = [
        {
            "ticker": "BTC-USD",
            "signal": "BUY",
            "spot_price": 65000.0,
            "confidence": 0.85,
            "regime": "BULLISH",
            "microstructure_stop_loss": 62000.0,
            "tp1_target": 70000.0,
            "tp2_target": 75000.0,
        }
    ]

    res = broker.execute_daily_signals(signals)
    assert len(res["buys"]) == 1
    btc_pos = broker.state["open_positions"]["BTC-USD"]
    # Shares should be a float with fractional quantity, NOT 0!
    assert btc_pos["shares"] > 0
    assert isinstance(btc_pos["shares"], float)
    assert btc_pos["shares"] < 1.0  # Fraction of 1 BTC
