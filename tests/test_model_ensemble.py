import pytest
from src.model_ensemble import (
    blend_model_predictions,
    calculate_triple_barrier_corridors,
)


def test_blend_model_predictions_consensus_bullish():
    """Verify that high agreement between XGBoost and Deep Learning boosts conviction."""
    res = blend_model_predictions(
        xgb_prob=0.70,
        dl_prob=0.72,
        sentiment_score=0.65,
    )
    assert res["signal"] == "BUY"
    assert res["consensus_state"] == "CONVICTION_AGREEMENT_BULLISH"
    assert res["ensemble_probability"] >= 0.70


def test_blend_model_predictions_divergence_neutralized():
    """Verify that conflict between models blocks the trade and returns NEUTRAL."""
    res = blend_model_predictions(
        xgb_prob=0.75,  # XGBoost says Bullish
        dl_prob=0.30,  # Deep Learning says Bearish
    )
    assert res["signal"] == "NEUTRAL"
    assert res["consensus_state"] == "MODEL_DIVERGENCE_NEUTRALIZED"
    assert res["ensemble_probability"] == 0.50


def test_calculate_triple_barrier_corridors():
    """Verify ATR-based multi-stage profit and stop corridors."""
    corridors = calculate_triple_barrier_corridors(
        entry_price=100.0,
        atr=2.5,
        tp1_multiplier=2.0,
        tp2_multiplier=4.0,
        sl_multiplier=1.0,
    )
    assert corridors["take_profit_1"] == 105.0  # 100 + 2*2.5
    assert corridors["take_profit_2"] == 110.0  # 100 + 4*2.5
    assert corridors["stop_loss"] == 97.5  # 100 - 1*2.5
    assert corridors["risk_reward_ratio"] == 2.0
