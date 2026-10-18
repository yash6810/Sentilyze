"""
Test suite for the new Quantitative Alpha Modules:
1. Overnight Drift Anomaly Engine (Cooper, Dong & Vines 2008 / Lou, Polk & Skouras 2019)
2. FINRA RegSho Short Volume & Short Squeeze Radar
3. Google TimesFM Quantile Price Corridors (p10, p50, p90)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.overnight_drift import (
    compute_overnight_drift_metrics,
    OvernightDriftEngine,
    is_moc_entry_window,
    is_open_exit_window,
)
from src.short_squeeze_radar import (
    estimate_short_volume_ratio,
    ShortSqueezeRadar,
)
from src.timesfm_corridors import TimesFMCorridorEngine


def test_overnight_drift_metrics():
    # Synthetic dataframe with positive overnight drift
    dates = pd.date_range("2026-08-01", periods=30, freq="B")
    close_vals = [100.0 + i * 0.5 for i in range(30)]
    open_vals = [c + 0.3 for c in close_vals]  # Overnight gaps higher

    df = pd.DataFrame(
        {
            "Open": open_vals,
            "High": [o + 1.0 for o in open_vals],
            "Low": [c - 0.5 for c in close_vals],
            "Close": close_vals,
            "Volume": [1000000] * 30,
        },
        index=dates,
    )

    metrics = compute_overnight_drift_metrics(df)
    assert "overnight_mean_return_pct" in metrics
    assert "overnight_ir" in metrics
    assert metrics["overnight_win_rate"] >= 50.0
    assert metrics["recommendation"] in [
        "STRONG_OVERNIGHT_EDGE",
        "MODERATE_OVERNIGHT_EDGE",
    ]


def test_overnight_drift_engine():
    engine = OvernightDriftEngine()
    res = engine.evaluate_ticker("NVDA", spot_price=135.0)
    assert "ticker" in res
    assert res["ticker"] == "NVDA"
    assert "expected_open_target" in res
    assert "protective_stop" in res
    assert res["protective_stop"] < res["spot_price"]


def test_short_squeeze_radar():
    dates = pd.date_range("2026-08-01", periods=25, freq="B")
    df = pd.DataFrame(
        {
            "Open": [50.0 + i * 0.4 for i in range(25)],
            "High": [52.0 + i * 0.4 for i in range(25)],
            "Low": [49.5 + i * 0.4 for i in range(25)],
            "Close": [51.5 + i * 0.4 for i in range(25)],
            "Volume": [2000000] * 25,
        },
        index=dates,
    )

    ratio_info = estimate_short_volume_ratio(df)
    assert 0.20 <= ratio_info["short_volume_ratio"] <= 0.80

    radar = ShortSqueezeRadar()
    analysis = radar.analyze_ticker("TEST_TICKER", spot_price=65.0)
    assert "squeeze_score" in analysis
    assert 0.0 <= analysis["squeeze_score"] <= 100.0
    assert "verdict" in analysis


def test_timesfm_quantile_corridors():
    engine = TimesFMCorridorEngine(horizon_days=5)
    res = engine.generate_corridors("NVDA", spot_price=140.0)

    assert res["ticker"] == "NVDA"
    assert res["horizon_days"] == 5
    assert len(res["p10_corridor"]) == 5
    assert len(res["p50_corridor"]) == 5
    assert len(res["p90_corridor"]) == 5

    # Quantile monotonicity guarantee: p10 <= p50 <= p90 at all forecast steps
    for i in range(5):
        p10 = res["p10_corridor"][i]
        p50 = res["p50_corridor"][i]
        p90 = res["p90_corridor"][i]
        assert (
            p10 <= p50 <= p90
        ), f"Quantile ordering violated at step {i}: {p10} <= {p50} <= {p90}"

    assert res["timesfm_stop_loss"] <= res["spot_price"]
    assert res["timesfm_tp1_target"] >= res["timesfm_stop_loss"]
    assert res["timesfm_tp2_runner"] >= res["timesfm_tp1_target"]
    assert res["corridor_width_pct"] > 0.0
