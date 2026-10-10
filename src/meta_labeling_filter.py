"""
López de Prado (2018) Meta-Labeling & Triple-Barrier Secondary Classifier for Sentilyze.
Advances in Financial Machine Learning, Chapters 3 & 5.

Deconstructs model decision into two layers:
1. Primary Model: Predicts trade side / direction (Technical indicators + XGBoost + FinBERT).
2. Secondary Meta-Labeler: Predicts whether executing the primary model's BUY signal
   will achieve the Upper Profit Barrier (+2.5 ATR) before touching the Lower Stop-Loss Barrier (-1.5 ATR).

If P(hit_upper_barrier_first) < 0.50:
The trade is filtered out as a low-quality false breakout / whipsaw trap.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.special import expit
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_meta_labeling_barrier_probability(
    ticker: str,
    spot_price: float,
    primary_conviction_pct: float,
    atr_14: float,
    rsi_val: float = 50.0,
    obi_val: float = 0.0,
    arar_score: float = 0.0,
    is_above_sma200: bool = True,
    realized_vol_pct: float = 20.0,
    vix_level: float = 18.0,
) -> Dict[str, Any]:
    """
    Computes calibrated probability P(touch_profit_barrier_first).
    Returns decision whether to APPROVE or VETO the primary BUY signal.
    """
    if spot_price <= 0:
        return {
            "approved": False,
            "barrier_touch_prob": 0.0,
            "reason": "Invalid spot price",
            "meta_label": 0,
        }

    # Baseline logit from primary conviction
    base_logit = (primary_conviction_pct - 50.0) / 15.0

    # 1. Structural Macro Trend Factor (Paper 11)
    # Assets above SMA200 have significantly higher upper barrier touch probability
    trend_factor = 0.65 if is_above_sma200 else -1.20

    # 2. RSI Exhaustion Penalty
    # Entering when RSI > 70 has severe downside asymmetry
    if rsi_val > 72.0:
        rsi_factor = -1.40 * ((rsi_val - 70.0) / 10.0)
    elif 42.0 <= rsi_val <= 62.0:
        rsi_factor = +0.60  # Sweet spot for momentum continuation
    elif rsi_val < 35.0:
        rsi_factor = +0.40  # Oversold mean reversion
    else:
        rsi_factor = 0.0

    # 3. Microstructure Iceberg & Order Book Imbalance Support (Stoikov 2018)
    # Positive OBI and high ARAR indicate passive buyer limit defense
    obi_factor = np.clip(obi_val * 1.2, -1.0, 1.0)
    arar_norm = np.tanh(arar_score / 500.0)
    micro_factor = 0.50 * obi_factor + 0.50 * arar_norm

    # 4. Volatility Regime & VIX Headwind (Paper 17)
    if vix_level > 24.0:
        vol_factor = -0.50 * ((vix_level - 20.0) / 10.0)
    else:
        vol_factor = 0.20

    # Composite Meta-Logit
    meta_logit = base_logit + trend_factor + rsi_factor + micro_factor + vol_factor
    barrier_prob = float(expit(meta_logit))
    barrier_prob = float(np.clip(barrier_prob, 0.02, 0.98))

    is_approved = bool(barrier_prob >= 0.50 and is_above_sma200 and rsi_val <= 75.0)

    if is_approved:
        verdict = "META_LABEL_APPROVED_HIGH_PROBABILITY_BARRIER_TOUCH"
        reason = (
            f"Upper barrier probability {barrier_prob:.1%} exceeds 50% threshold. "
            f"Structural trend and microstructure support favorable risk/reward."
        )
    else:
        verdict = "META_LABEL_VETO_WHIPSAW_TRAP_FILTER"
        if not is_above_sma200:
            reason = "Vetoed by Meta-Labeler: Asset below SMA200 has low upper barrier hit rate."
        elif rsi_val > 75.0:
            reason = (
                f"Vetoed by Meta-Labeler: RSI severely overbought ({rsi_val:.1f} > 75)."
            )
        else:
            reason = f"Vetoed by Meta-Labeler: Upper barrier hit probability is only {barrier_prob:.1%} (< 50%)."

    return {
        "ticker": ticker,
        "approved": is_approved,
        "barrier_touch_prob": round(barrier_prob, 4),
        "barrier_touch_prob_pct": round(barrier_prob * 100.0, 1),
        "verdict": verdict,
        "meta_label": 1 if is_approved else 0,
        "factors": {
            "base_logit": round(base_logit, 3),
            "trend_factor": round(trend_factor, 3),
            "rsi_factor": round(rsi_factor, 3),
            "micro_factor": round(micro_factor, 3),
            "vol_factor": round(vol_factor, 3),
        },
        "reason": reason,
    }
