import numpy as np
import pandas as pd
from src.temporal_fusion import (
    ScaledDotProductAttention,
    TemporalFusionEngine,
    run_temporal_fusion_forecast,
)


def test_scaled_dot_product_attention():
    attn = ScaledDotProductAttention(d_k=16)
    Q = np.random.randn(20, 16)
    K = np.random.randn(20, 16)
    V = np.random.randn(20, 16)

    context, weights = attn.forward(Q, K, V)
    assert context.shape == (20, 16)
    assert weights.shape == (20, 20)
    # Each row of weights should sum to 1.0 (softmax)
    row_sums = np.sum(weights, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_temporal_fusion_engine_multihorizon():
    tft = TemporalFusionEngine(lookback_window=25, feature_dim=6)
    feat_mat = np.random.randn(25, 6)
    res = tft.forecast_multihorizon(feat_mat, current_price=150.0)

    assert "horizons" in res
    assert "1_day" in res["horizons"]
    assert "5_days" in res["horizons"]
    assert "10_days" in res["horizons"]
    assert "21_days" in res["horizons"]

    h1 = res["horizons"]["1_day"]
    assert "q10_bear" in h1
    assert "q50_median" in h1
    assert "q90_bull" in h1
    assert h1["q10_bear"] <= h1["q50_median"] <= h1["q90_bull"]

    assert len(res["temporal_attention_weights"]) == 25
    assert len(res["feature_importance_weights"]) == 6


def test_run_temporal_fusion_forecast():
    df = pd.DataFrame(np.random.randn(50, 6), columns=[f"feat_{i}" for i in range(6)])
    res = run_temporal_fusion_forecast("NVDA", df, current_price=200.0)

    assert res["ticker"] == "NVDA"
    assert res["current_price"] == 200.0
    assert "horizons" in res
