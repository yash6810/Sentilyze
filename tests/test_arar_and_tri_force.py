"""
Unit tests for Institutional ARAR Microstructure Engine & Tri-Force Dynamic Signal Fusion.
Validates:
1. Level-1 Order Book Imbalance (OBI) calculations.
2. Bulk Volume Classification (BVC) Cumulative Volume Delta (CVD).
3. ARAR absorption ratio spikes during passive iceberg wall defense.
4. Tri-Force dynamic regime weights shifting under high volatility vs chop vs trend.
5. Tri-Force composite trade probability P_trade calibration.
6. Marcos López de Prado Purged K-Fold Cross-Validation & Deflated Sharpe Ratio (DSR).
"""

import numpy as np
import pandas as pd
import pytest

from src.arar_microstructure import (
    calculate_order_book_imbalance,
    calculate_bar_cumulative_volume_delta,
    compute_arar_absorption_ratio,
    evaluate_arar_microstructure,
)
from src.tri_force_fusion import (
    compute_hurst_exponent,
    calculate_tri_force_regime_weights,
    calculate_statarb_signal_from_series,
    evaluate_tri_force_fusion,
    PurgedKFold,
    calculate_deflated_sharpe_ratio,
)


def test_order_book_imbalance_calculation():
    # Massive bid wall
    obi_bid = calculate_order_book_imbalance(bid_volume=100000, ask_volume=10000)
    assert obi_bid > 0.80

    # Massive ask overhang
    obi_ask = calculate_order_book_imbalance(bid_volume=5000, ask_volume=95000)
    assert obi_ask < -0.85

    # Balanced book
    obi_neutral = calculate_order_book_imbalance(bid_volume=50000, ask_volume=50000)
    assert obi_neutral == 0.0


def test_cumulative_volume_delta():
    # Create sample DataFrame
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    prices = [100.0 + i * 0.5 for i in range(30)]
    vols = [1000000 + (i % 3) * 200000 for i in range(30)]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1.0 for p in prices],
            "Low": [p - 0.5 for p in prices],
            "Close": prices,
            "Volume": vols,
        },
        index=dates,
    )

    cvd_df = calculate_bar_cumulative_volume_delta(df)
    assert "cvd" in cvd_df.columns
    assert "volume_delta" in cvd_df.columns
    assert len(cvd_df) == len(df)


def test_arar_absorption_ratio_math():
    # Aggressive selling dumped into immovable bid wall
    # High selling volume, high positive OBI, zero price movement
    arar_wall = compute_arar_absorption_ratio(
        cvd_sell=500000.0,
        obi_bid=0.95,
        delta_price=0.001,
        epsilon=0.01,
    )
    assert arar_wall > 40000000.0

    # Large price breakdown (wall failed, high delta price)
    arar_failed = compute_arar_absorption_ratio(
        cvd_sell=500000.0,
        obi_bid=0.95,
        delta_price=5.0,
        epsilon=0.01,
    )
    assert arar_failed < arar_wall / 100.0


def test_evaluate_arar_microstructure():
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    prices = [150.0 + np.sin(i / 3.0) * 3.0 for i in range(25)]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1.5 for p in prices],
            "Low": [p - 1.2 for p in prices],
            "Close": prices,
            "Volume": [1500000 for _ in range(25)],
        },
        index=dates,
    )

    res = evaluate_arar_microstructure("NVDA", df)
    assert "arar_score" in res
    assert "microstructure_signal" in res
    assert -1.0 <= res["microstructure_signal"] <= 1.0


def test_tri_force_regime_weight_adaptation():
    # 1. High Volatility Regime: Microstructure should dominate
    w_mu_vol, w_arb_vol, w_trend_vol, reg_vol = calculate_tri_force_regime_weights(
        realized_vol_pct=45.0,
        hurst_exponent=0.50,
        vix_level=32.0,
        trend_adx=20.0,
    )
    assert w_mu_vol > w_arb_vol
    assert w_mu_vol > w_trend_vol
    assert np.isclose(w_mu_vol + w_arb_vol + w_trend_vol, 1.0, atol=1e-3)
    assert reg_vol == "HIGH_VOLATILITY_MICROSTRUCTURE_DOMINANT"

    # 2. Choppy Sideways Regime: Stat-Arb should dominate
    w_mu_chop, w_arb_chop, w_trend_chop, reg_chop = calculate_tri_force_regime_weights(
        realized_vol_pct=14.0,
        hurst_exponent=0.35,  # Strong mean-reversion
        vix_level=13.0,
        trend_adx=12.0,
    )
    assert w_arb_chop > w_mu_chop
    assert w_arb_chop > w_trend_chop
    assert reg_chop == "SIDEWAYS_CHOP_STATARB_DOMINANT"

    # 3. Momentum Trending Regime: Trend should dominate
    w_mu_tr, w_arb_tr, w_trend_tr, reg_tr = calculate_tri_force_regime_weights(
        realized_vol_pct=22.0,
        hurst_exponent=0.72,  # Strong trend persistence
        vix_level=17.0,
        trend_adx=38.0,
    )
    assert w_trend_tr > w_mu_tr
    assert w_trend_tr > w_arb_tr
    assert reg_tr == "MOMENTUM_TREND_DOMINANT"


def test_evaluate_tri_force_fusion():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    prices = [200.0 + i * 1.5 for i in range(40)]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 2.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": [2000000 for _ in range(40)],
        },
        index=dates,
    )

    fusion = evaluate_tri_force_fusion(
        ticker="NVDA",
        df_ohlcv=df,
        xgb_prob=0.80,
        finbert_sentiment=0.65,
        vix_level=18.0,
    )

    assert "p_trade" in fusion
    assert 0.0 <= fusion["p_trade"] <= 1.0
    assert "weights" in fusion
    assert "signals" in fusion
    assert fusion["p_trade"] >= 0.60  # Strong bullish inputs


def test_purged_kfold_cross_validation():
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    X = pd.DataFrame({"feature": np.random.randn(100)}, index=dates)

    pkf = PurgedKFold(n_splits=5, pct_embargo=0.02)
    splits = list(pkf.split(X))
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        assert len(test_idx) == 20
        # Check no overlap between train and test
        overlap = set(train_idx).intersection(set(test_idx))
        assert len(overlap) == 0


def test_deflated_sharpe_ratio():
    # Solid Sharpe with low trials -> high DSR
    dsr_high = calculate_deflated_sharpe_ratio(
        sharpe_est=2.5,
        n_trials=5,
        var_sharpe=0.5,
    )
    assert dsr_high > 0.80

    # Modest Sharpe tested across 10,000 trials (data snooping) -> low DSR
    dsr_overfit = calculate_deflated_sharpe_ratio(
        sharpe_est=1.1,
        n_trials=10000,
        var_sharpe=1.0,
    )
    assert dsr_overfit < dsr_high
