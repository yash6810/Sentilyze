import pytest
import numpy as np
import pandas as pd
from src.benchmark_training_paradigms import (
    run_paradigm_rolling_wfo,
    run_paradigm_expanding_timedecay,
    run_paradigm_triple_barrier_meta,
    run_paradigm_mixture_of_experts,
    run_paradigm_online_continual,
    run_paradigm_cross_asset_pooled,
    run_paradigm_direct_reinforcement_policy,
    run_paradigm_conformal_calibrated,
)


@pytest.fixture
def sample_synthetic_dataset():
    """Generates synthetic dataset for rapid testing of all 8 paradigms."""
    np.random.seed(42)
    n_samples = 600
    dates = pd.date_range("2022-01-01", periods=n_samples, freq="D")

    # Synthetic prices
    returns = np.random.normal(0.0005, 0.015, n_samples)
    price = 100 * np.exp(np.cumsum(returns))

    df_raw = pd.DataFrame(
        {
            "Open": price * (1 + np.random.normal(0, 0.002, n_samples)),
            "High": price * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            "Low": price * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            "Close": price,
            "Volume": np.random.randint(100000, 5000000, n_samples),
            "SMA200": pd.Series(price).rolling(200).mean().bfill(),
            "RSI": pd.Series(50 + 20 * np.random.randn(n_samples)).clip(10, 90),
            "ATR": pd.Series(price * 0.02),
        },
        index=dates,
    )

    # Features
    X = pd.DataFrame(
        {
            "feature_rsi": df_raw["RSI"],
            "feature_atr": df_raw["ATR"],
            "feature_ret5": pd.Series(returns).rolling(5).mean().fillna(0),
            "feature_vol": np.log(df_raw["Volume"]),
        },
        index=dates,
    )

    # Binary Target (1 = Up, 0 = Down)
    y = pd.Series((np.roll(returns, -1) > 0).astype(int), index=dates)
    y.iloc[-1] = 0

    return df_raw, X, y


def test_all_8_training_paradigms(sample_synthetic_dataset):
    df_raw, X, y = sample_synthetic_dataset
    train_w = 400
    test_w = 20

    # 1. Rolling WFO
    p1, m1 = run_paradigm_rolling_wfo(X, y, train_window=train_w, test_window=test_w)
    assert len(p1) > 0
    assert 0.0 <= m1["auc"] <= 1.0

    # 2. Expanding Window + Exponential Decay
    p2, m2 = run_paradigm_expanding_timedecay(
        X, y, min_train_window=train_w, test_window=test_w
    )
    assert len(p2) > 0
    assert 0.0 <= m2["auc"] <= 1.0

    # 3. Triple-Barrier Meta-Labeling
    p3, m3 = run_paradigm_triple_barrier_meta(
        df_raw, X, y, train_window=train_w, test_window=test_w
    )
    assert len(p3) > 0
    assert 0.0 <= m3["auc"] <= 1.0

    # 4. Mixture of Experts
    p4, m4 = run_paradigm_mixture_of_experts(
        df_raw, X, y, train_window=train_w, test_window=test_w
    )
    assert len(p4) > 0
    assert 0.0 <= m4["auc"] <= 1.0

    # 5. Online Continual Streaming
    p5, m5 = run_paradigm_online_continual(X, y, warmup_days=train_w)
    assert len(p5) > 0
    assert 0.0 <= m5["auc"] <= 1.0

    # 6. Cross-Asset Pooled Multi-Task
    asset_map = {"SYNTH_A": (df_raw, X, y), "SYNTH_B": (df_raw, X, y)}
    p6, m6 = run_paradigm_cross_asset_pooled(
        asset_map, "SYNTH_A", train_window=train_w, test_window=test_w
    )
    assert len(p6) > 0
    assert 0.0 <= m6["auc"] <= 1.0

    # 7. Direct Reinforcement Policy
    p7, m7 = run_paradigm_direct_reinforcement_policy(
        df_raw, X, y, train_window=train_w, test_window=test_w
    )
    assert len(p7) > 0
    assert 0.0 <= m7["auc"] <= 1.0

    # 8. Conformal Calibrated Uncertainty
    p8, m8 = run_paradigm_conformal_calibrated(
        X, y, train_window=train_w, test_window=test_w
    )
    assert len(p8) > 0
    assert 0.0 <= m8["auc"] <= 1.0
