"""
Unit tests for Market Microstructure Engine (Sprint 1, Modules 1.5 & 1.6)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, time

from src.microstructure_engine import (
    calculate_stoikov_micro_price,
    calculate_bar_stoikov_micro_price,
    calculate_bvc_flow_toxicity,
    TimeOfDayVolumeNormalizer,
    evaluate_microstructure_telemetry,
)


def test_calculate_stoikov_micro_price():
    # If bid size > ask size (buyer imbalance), micro price should skew towards ask
    bid_p, ask_p = 100.0, 100.20
    bid_q, ask_q = 800, 200  # heavy bid
    p_micro = calculate_stoikov_micro_price(bid_p, ask_p, bid_q, ask_q)
    mid = 100.10
    assert p_micro > mid
    assert p_micro <= ask_p

    # If ask size > bid size (seller imbalance), micro price should skew towards bid
    bid_q, ask_q = 100, 900
    p_micro_down = calculate_stoikov_micro_price(bid_p, ask_p, bid_q, ask_q)
    assert p_micro_down < mid
    assert p_micro_down >= bid_p


def test_calculate_bar_stoikov_micro_price():
    df = pd.DataFrame(
        {
            "High": [105.0, 106.0, 107.0],
            "Low": [100.0, 101.0, 102.0],
            "Close": [
                104.5,
                101.5,
                106.8,
            ],  # bar 0 closes near high, bar 1 near low, bar 2 near high
            "Volume": [1000, 1500, 2000],
        }
    )
    res = calculate_bar_stoikov_micro_price(df)
    assert "stoikov_micro_price" in res.columns
    assert "micro_price_delta_pct" in res.columns
    # Bar 0 closed near high, micro price delta pct should be positive
    assert res["micro_price_delta_pct"].iloc[0] > 0
    # Bar 1 closed near low, micro price delta pct should be negative
    assert res["micro_price_delta_pct"].iloc[1] < 0


def test_calculate_bvc_flow_toxicity():
    np.random.seed(42)
    closes = [100.0]
    for _ in range(30):
        closes.append(closes[-1] + np.random.normal(0, 1))
    vols = [1000 + i * 50 for i in range(len(closes))]

    df = pd.DataFrame({"Close": closes, "Volume": vols})
    res = calculate_bvc_flow_toxicity(df)
    assert "bvc_buy_volume" in res.columns
    assert "bvc_sell_volume" in res.columns
    assert "bvc_toxicity" in res.columns
    assert "bvc_cumulative_delta" in res.columns
    assert (res["bvc_toxicity"] >= 0.0).all() and (res["bvc_toxicity"] <= 1.0).all()


def test_time_of_day_volume_normalizer():
    norm = TimeOfDayVolumeNormalizer()
    morning_time = datetime(2026, 9, 23, 9, 35)
    lunch_time = datetime(2026, 9, 23, 12, 15)

    res_morning = norm.normalize_volume(
        raw_volume=200000, timestamp=morning_time, average_bucket_volume=100000
    )
    res_lunch = norm.normalize_volume(
        raw_volume=200000, timestamp=lunch_time, average_bucket_volume=100000
    )

    # 200k at 9:35 AM is expected (multiplier ~2.45), so RVOL should be ~0.8
    assert res_morning["u_curve_multiplier"] > 2.0
    # 200k at 12:15 PM lunch is massive surge (multiplier ~0.60), so RVOL should be > 2.5
    assert res_lunch["normalized_rvol"] > res_morning["normalized_rvol"]
    assert res_lunch["is_breakout_confirmed"] is True


def test_evaluate_microstructure_telemetry():
    # Smoke test on ticker
    telem = evaluate_microstructure_telemetry("NVDA", spot_price=220.0)
    assert "status" in telem
    assert "stoikov_micro_price" in telem
    assert "bvc_toxicity" in telem
    assert "kyles_lambda" in telem
    assert "microstructure_verdict" in telem


def test_calculate_kyles_lambda():
    from src.microstructure_engine import calculate_kyles_lambda

    df = pd.DataFrame(
        {
            "Close": [100.0 + i * 0.5 for i in range(20)],
            "Volume": [1000 + i * 100 for i in range(20)],
        }
    )
    kyle = calculate_kyles_lambda(df)
    assert "kyles_lambda" in kyle
    assert "implied_market_depth_dollars" in kyle
    assert kyle["kyles_lambda"] >= 0.0


def test_detect_iceberg_order_tape():
    from src.microstructure_engine import detect_iceberg_order_tape

    # Displayed 500, but executed 5000 -> 10x hidden iceberg
    res = detect_iceberg_order_tape(
        displayed_depth=500, executed_tape_volume=5000, price_level=150.0, is_bid=True
    )
    assert res["is_iceberg_detected"] is True
    assert res["hidden_multiplier"] == 10.0
    assert "ICEBERG_BID_ACCUMULATION" in res["institutional_signal"]

    # Normal order
    res_norm = detect_iceberg_order_tape(
        displayed_depth=1000, executed_tape_volume=500, price_level=150.0
    )
    assert res_norm["is_iceberg_detected"] is False


def test_calculate_order_book_resilience():
    from src.microstructure_engine import calculate_order_book_resilience

    # Fast full replenishment
    res_good = calculate_order_book_resilience(
        initial_depth=10000,
        depth_post_sweep=1000,
        depth_recovered=9000,
        recovery_time_seconds=1.5,
    )
    assert res_good["is_genuine_replenishment"] is True
    assert res_good["is_fake_liquidity_wall"] is False

    # Fake wall that failed to replenish
    res_fake = calculate_order_book_resilience(
        initial_depth=10000,
        depth_post_sweep=1000,
        depth_recovered=2000,
        recovery_time_seconds=8.0,
    )
    assert res_fake["is_fake_liquidity_wall"] is True


def test_calculate_l2_depth_weighted_imbalance():
    from src.microstructure_engine import calculate_l2_depth_weighted_imbalance

    bids = [(100.0, 5000), (99.9, 4000), (99.8, 3000)]
    asks = [(100.1, 1000), (100.2, 1000), (100.3, 1000)]
    res = calculate_l2_depth_weighted_imbalance(bids, asks)
    assert res["l2_imbalance"] > 0.40
    assert res["imbalance_regime"] == "STRONG_BID_SUPPORT_STACKING"
