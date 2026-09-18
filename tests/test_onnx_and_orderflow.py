"""
Unit tests for ONNX FinBERT Acceleration & Order Flow Candlestick Engine.
========================================================================
Verifies:
1. ONNX FinBERT Engine inference, caching, and fallback resilience.
2. Volume Profile (VPVR), Point of Control (PoC), and Value Area calculation.
3. Fair Value Gap (FVG) detection.
4. Plotly dual-pane Order Flow chart builder integrity.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.orderflow_chart import (
    calculate_volume_profile,
    detect_fair_value_gaps,
    build_orderflow_candlestick_chart,
)
from src.onnx_finbert import ONNXFinBERTEngine


@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    np.random.seed(42)
    closes = 100.0 + np.cumsum(np.random.randn(30) * 2.0)
    highs = closes + np.random.uniform(0.5, 3.0, 30)
    lows = closes - np.random.uniform(0.5, 3.0, 30)
    opens = closes + np.random.uniform(-1.0, 1.0, 30)
    volumes = np.random.uniform(100_000, 1_000_000, 30)

    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
            "sma50": closes * 0.98,
            "sma200": closes * 0.95,
        },
        index=dates,
    )


def test_calculate_volume_profile(sample_ohlcv_df):
    bins, poc_price, vah_price, val_price = calculate_volume_profile(
        sample_ohlcv_df, n_bins=20
    )
    assert len(bins) == 20
    min_p = sample_ohlcv_df["Low"].min()
    max_p = sample_ohlcv_df["High"].max()

    assert min_p <= poc_price <= max_p
    assert min_p <= val_price <= vah_price <= max_p

    poc_bins = [b for b in bins if b["is_poc"]]
    assert len(poc_bins) == 1
    assert poc_bins[0]["rel_volume"] == 1.0


def test_detect_fair_value_gaps():
    # Construct synthetic candles with a clear Bullish FVG
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 106.0, 110.0, 111.0],
            "High": [101.0, 105.0, 109.0, 112.0, 113.0],  # Candle 0 High = 101.0
            "Low": [
                99.0,
                101.5,
                104.0,
                108.0,
                110.0,
            ],  # Candle 2 Low = 104.0 (> 101.0 -> FVG)
            "Close": [100.5, 104.5, 108.5, 111.5, 112.0],
            "Volume": [1000] * 5,
        },
        index=dates,
    )
    fvgs = detect_fair_value_gaps(df)
    assert len(fvgs) >= 1
    bull_fvg = fvgs[0]
    assert bull_fvg["type"] == "BULLISH_FVG"
    assert bull_fvg["bottom"] == 101.0  # Candle 0 High
    assert bull_fvg["top"] == 104.0  # Candle 2 Low


def test_build_orderflow_candlestick_chart(sample_ohlcv_df):
    orb_levels = {"orb_high": 115.0, "orb_low": 95.0}
    fig = build_orderflow_candlestick_chart(
        sample_ohlcv_df,
        ticker="NVDA",
        orb_levels=orb_levels,
        height=600,
    )

    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if hasattr(t, "name")]
    assert "Price" in trace_names
    assert "Volume" in trace_names
    assert "Vol MA20" in trace_names


def test_build_orderflow_candlestick_chart_empty_graceful():
    empty_df = pd.DataFrame()
    fig = build_orderflow_candlestick_chart(empty_df, ticker="EMPTY")
    assert isinstance(fig, go.Figure)


def test_onnx_finbert_engine_prediction_and_caching():
    engine = ONNXFinBERTEngine(force_fallback=True)
    assert engine.is_onnx_active is False

    headline = "NVIDIA reports exceptional data center quarterly revenue growth"
    res = engine.predict_single(headline, ticker="NVDA")

    assert "sentiment_label" in res
    assert "confidence" in res
    assert "sentiment_score" in res
    assert "latency_ms" in res
    assert res["engine"] in ["ONNX_RUNTIME", "PYTORCH_FALLBACK", "SAFE_DEFAULT"]

    # Test cache retrieval
    res_cached = engine.predict_single(headline, ticker="NVDA")
    assert res_cached.get("cached") is True
    assert res_cached["sentiment_label"] == res["sentiment_label"]


def test_onnx_finbert_empty_text():
    engine = ONNXFinBERTEngine(force_fallback=True)
    res = engine.predict_single("")
    assert res["sentiment_label"] == "neutral"
    assert res["confidence"] == 0.5
    assert res["sentiment_score"] == 0.0


def test_onnx_finbert_batch():
    engine = ONNXFinBERTEngine(force_fallback=True)
    texts = ["Stock surges higher", "Company files bankruptcy"]
    batch_res = engine.predict_batch(texts)
    assert len(batch_res) == 2
    assert batch_res[0]["sentiment_label"] in ["positive", "neutral", "negative"]
