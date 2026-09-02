"""
Dual-Model Consensus Alpha Engine: XGBoost + DLinear-TCN Deep Learning Fusion.
Combines tabular gradient-boosted decision trees with temporal deep learning sequence trajectories.

Ensemble Logic:
1. High-Conviction Consensus: Triggers trade execution only when XGBoost AND Deep Learning agree.
2. Conflict Neutralization: If models diverge, the trade is blocked (protects capital from false breakouts).
3. Triple-Barrier Risk Sizing: Targets +2.0 ATR (TP1), +4.0 ATR (TP2), and -1.0 ATR (SL).
"""

import numpy as np
from typing import Dict, Any, Optional

from src.utils import get_logger

logger = get_logger(__name__)


def blend_model_predictions(
    xgb_prob: float,
    dl_prob: float,
    sentiment_score: float = 0.5,
    weight_xgb: float = 0.50,
    weight_dl: float = 0.50,
    min_consensus_threshold: float = 0.53,
) -> Dict[str, Any]:
    """
    Blends XGBoost and Deep Learning probabilities with consensus-gated execution.

    Args:
      - xgb_prob: Probability of positive return from XGBoost [0.0, 1.0]
      - dl_prob: Probability of positive return from DLinear-TCN Deep Learning [0.0, 1.0]
      - sentiment_score: FinBERT / DistilRoBERTa polarity score [0.0, 1.0]
      - weight_xgb: Weight for XGBoost probability
      - weight_dl: Weight for Deep Learning probability
      - min_consensus_threshold: Threshold required from both models to confirm trade

    Returns:
      Dict with ensemble_probability, consensus_state, signal, confidence, and thesis.
    """
    # Linear weighted blend
    raw_blend = (xgb_prob * weight_xgb) + (dl_prob * weight_dl)

    # Consensus Check
    both_bullish = (
        xgb_prob >= min_consensus_threshold and dl_prob >= min_consensus_threshold
    )
    both_bearish = xgb_prob <= (1.0 - min_consensus_threshold) and dl_prob <= (
        1.0 - min_consensus_threshold
    )

    if both_bullish:
        consensus_state = "CONVICTION_AGREEMENT_BULLISH"
        final_prob = min(0.95, raw_blend + 0.05)  # Synergistic boost
        signal = "BUY"
    elif both_bearish:
        consensus_state = "CONVICTION_AGREEMENT_BEARISH"
        final_prob = max(0.05, raw_blend - 0.05)
        signal = "SELL"
    else:
        consensus_state = "MODEL_DIVERGENCE_NEUTRALIZED"
        # Conflict between tree logic and neural sequence -> Neutralize to avoid bad trades
        final_prob = 0.50
        signal = "NEUTRAL"

    confidence = round(abs(final_prob - 0.5) * 2.0, 4)

    thesis = (
        f"XGBoost ({xgb_prob:.1%}) & Deep Learning ({dl_prob:.1%}) "
        f"reached {consensus_state.replace('_', ' ').title()}."
    )

    return {
        "ensemble_probability": round(float(final_prob), 4),
        "raw_blend": round(float(raw_blend), 4),
        "xgb_probability": round(float(xgb_prob), 4),
        "dl_probability": round(float(dl_prob), 4),
        "sentiment_score": round(float(sentiment_score), 4),
        "consensus_state": consensus_state,
        "signal": signal,
        "confidence": confidence,
        "thesis": thesis,
    }


def calculate_triple_barrier_corridors(
    entry_price: float,
    atr: float,
    tp1_multiplier: float = 2.0,
    tp2_multiplier: float = 4.0,
    sl_multiplier: float = 1.0,
) -> Dict[str, float]:
    """
    Calculates dynamic institutional take-profit and stop-loss levels based on ATR volatility.
    """
    tp1 = entry_price + (tp1_multiplier * atr)
    tp2 = entry_price + (tp2_multiplier * atr)
    sl = entry_price - (sl_multiplier * atr)
    risk_reward = (tp1_multiplier) / (sl_multiplier + 1e-6)

    return {
        "entry_price": round(float(entry_price), 2),
        "take_profit_1": round(float(tp1), 2),
        "take_profit_2": round(float(tp2), 2),
        "stop_loss": round(float(sl), 2),
        "risk_reward_ratio": round(float(risk_reward), 2),
        "atr": round(float(atr), 2),
    }
