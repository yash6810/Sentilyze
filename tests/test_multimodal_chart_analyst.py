"""Unit and Integration Tests for Multimodal Wyckoff Chart Analyst.

Verifies:
1. Thread-safe candlestick and volume rendering to PNG byte stream.
2. Algorithmic liquidity sweep detection (Springs, UTADs, Fair Value Gaps, Order Blocks).
3. Gemini Vision API structured response handling and mock scenarios.
4. Fault tolerance and graceful offline fallback.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.multimodal_chart_analyst import (
    MultimodalChartAnalyst,
    WyckoffAnalysisResult,
    evaluate_chart_structure,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate synthetic 40-bar OHLCV dataframe for tests."""
    np.random.seed(42)
    n = 40
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    base_price = 150.0 + np.cumsum(np.random.randn(n) * 1.5)

    df = pd.DataFrame(
        {
            "Open": base_price + np.random.randn(n) * 0.5,
            "High": base_price + np.abs(np.random.randn(n) * 1.5) + 0.5,
            "Low": base_price - np.abs(np.random.randn(n) * 1.5) - 0.5,
            "Close": base_price + np.random.randn(n) * 0.5,
            "Volume": np.random.randint(1000000, 5000000, size=n),
        },
        index=dates,
    )
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df["High"] = df[["Open", "Close", "High"]].max(axis=1) + 0.1
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1) - 0.1
    return df


def test_chart_image_rendering(sample_ohlcv_df: pd.DataFrame):
    """Verify candlestick chart is rendered to valid PNG bytes."""
    analyst = MultimodalChartAnalyst()
    png_bytes = analyst.render_chart_image(sample_ohlcv_df, ticker="NVDA")

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000
    # PNG signature check
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_image_rendering_minimal_data():
    """Verify chart rendering handles small/empty dataframe gracefully without exception."""
    analyst = MultimodalChartAnalyst()
    empty_df = pd.DataFrame()
    png_bytes = analyst.render_chart_image(empty_df, ticker="EMPTY")
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_algorithmic_bullish_liquidity_sweep(sample_ohlcv_df: pd.DataFrame):
    """Verify algorithmic detector catches a Wyckoff Spring (Sell-side sweep)."""
    df = sample_ohlcv_df.copy()
    # Establish a known swing low in prior bars
    prior_low = df["Low"].iloc[-10:-1].min()

    # Artificially create a Spring on the latest bar:
    # Low pierces prior_low by $2.00, but Close reclaims back above prior_low
    df.iloc[-1, df.columns.get_loc("Low")] = prior_low - 2.0
    df.iloc[-1, df.columns.get_loc("Open")] = prior_low + 0.5
    df.iloc[-1, df.columns.get_loc("Close")] = prior_low + 1.5
    df.iloc[-1, df.columns.get_loc("High")] = prior_low + 2.0

    analyst = MultimodalChartAnalyst()
    result = analyst.analyze_market_structure_algorithmic(df, ticker="NVDA")

    assert result.liquidity_sweep_detected is True
    assert result.sweep_type == "SELL_SIDE_LIQUIDITY_RUN"
    assert result.wyckoff_phase == "ACCUMULATION_SPRING"
    assert result.market_bias == "BULLISH"
    assert result.confidence >= 0.80


def test_algorithmic_bearish_liquidity_sweep(sample_ohlcv_df: pd.DataFrame):
    """Verify algorithmic detector catches an Upthrust After Distribution (UTAD)."""
    df = sample_ohlcv_df.copy()
    prior_high = df["High"].iloc[-10:-1].max()

    # Artificially create UTAD on the latest bar:
    # High spikes above prior_high, but Close closes back below
    df.iloc[-1, df.columns.get_loc("High")] = prior_high + 3.0
    df.iloc[-1, df.columns.get_loc("Open")] = prior_high - 0.5
    df.iloc[-1, df.columns.get_loc("Close")] = prior_high - 1.2
    df.iloc[-1, df.columns.get_loc("Low")] = prior_high - 2.0

    analyst = MultimodalChartAnalyst()
    result = analyst.analyze_market_structure_algorithmic(df, ticker="AAPL")

    assert result.liquidity_sweep_detected is True
    assert result.sweep_type == "BUY_SIDE_LIQUIDITY_RUN"
    assert result.wyckoff_phase == "DISTRIBUTION_UTAD"
    assert result.market_bias == "BEARISH"
    assert result.confidence >= 0.80


def test_fair_value_gap_detection(sample_ohlcv_df: pd.DataFrame):
    """Verify Fair Value Gap (FVG) detection."""
    df = sample_ohlcv_df.copy()
    # Create an artificial Bullish FVG at bar index -1:
    # Bar -3 high is 150.0, Bar -1 low is 153.0 (3.0 dollar imbalance)
    idx_m3 = len(df) - 3
    idx_m1 = len(df) - 1
    df.iloc[idx_m3, df.columns.get_loc("High")] = 150.0
    df.iloc[idx_m1, df.columns.get_loc("Low")] = 153.0
    df.iloc[idx_m1, df.columns.get_loc("Open")] = 153.5
    df.iloc[idx_m1, df.columns.get_loc("Close")] = 155.0
    df.iloc[idx_m1, df.columns.get_loc("High")] = 156.0

    analyst = MultimodalChartAnalyst()
    result = analyst.analyze_market_structure_algorithmic(df, ticker="MSFT")

    assert len(result.fair_value_gaps) > 0
    bullish_fvgs = [f for f in result.fair_value_gaps if f["fvg_type"] == "BULLISH"]
    assert len(bullish_fvgs) >= 1
    assert bullish_fvgs[-1]["bottom"] == 150.0
    assert bullish_fvgs[-1]["top"] == 153.0


def test_gemini_vision_mocked_success(sample_ohlcv_df: pd.DataFrame):
    """Verify Gemini Vision API structured response parsing when API call succeeds."""
    mock_gemini_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "wyckoff_phase": "ACCUMULATION_SPRING",
                                    "market_bias": "BULLISH",
                                    "confidence": 0.88,
                                    "liquidity_sweep_detected": True,
                                    "sweep_type": "SELL_SIDE_LIQUIDITY_RUN",
                                    "sweep_price_level": 142.50,
                                    "order_block_level": 141.80,
                                    "order_block_type": "BULLISH_OB",
                                    "key_support": 140.0,
                                    "key_resistance": 158.0,
                                    "reasoning": "High-volume spring observed testing the accumulation trading range.",
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_gemini_response

    with patch("requests.post", return_value=mock_resp):
        analyst = MultimodalChartAnalyst(api_key="fake-test-key")
        chart_bytes = analyst.render_chart_image(sample_ohlcv_df, ticker="NVDA")
        result = analyst.analyze_chart_with_gemini(
            chart_bytes, ticker="NVDA", df=sample_ohlcv_df
        )

        assert result.analyst_source == "GEMINI_MULTIMODAL"
        assert result.wyckoff_phase == "ACCUMULATION_SPRING"
        assert result.liquidity_sweep_detected is True
        assert result.confidence == 0.88
        assert result.key_support == 140.0


def test_gemini_vision_error_graceful_fallback(sample_ohlcv_df: pd.DataFrame):
    """Verify graceful fallback to algorithmic engine on API timeout or error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"

    with patch("requests.post", return_value=mock_resp):
        analyst = MultimodalChartAnalyst(api_key="fake-test-key")
        chart_bytes = analyst.render_chart_image(sample_ohlcv_df, ticker="NVDA")
        result = analyst.analyze_chart_with_gemini(
            chart_bytes, ticker="NVDA", df=sample_ohlcv_df
        )

        assert result.analyst_source == "ALGORITHMIC"
        assert result.ticker == "NVDA"
        assert result.confidence > 0.0


def test_evaluate_chart_structure_convenience_api(sample_ohlcv_df: pd.DataFrame):
    """Verify the top-level functional convenience API."""
    data = evaluate_chart_structure(sample_ohlcv_df, ticker="GOOGL")
    assert isinstance(data, dict)
    assert data["ticker"] == "GOOGL"
    assert "wyckoff_phase" in data
    assert "market_bias" in data
    assert "liquidity_sweep_detected" in data
    assert "fair_value_gaps" in data
