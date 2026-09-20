"""
Retail Sentiment Peak & Short Squeeze Radar (SSPS) for Sentilyze.

Grounding & Mechanics:
1. Contrarian Sentiment Peak Exhaustion: Euphoric sentiment + lagging price momentum = exhaustion top warning.
2. Short Squeeze Probability Score (SSPS):
   SSPS = w1 * ShortInterestPct + w2 * DaysToCover + w3 * AbnormalVolumeRatio + w4 * SentimentVelocity
   Scores 0 to 100 with actionable gamma/squeeze alerts.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger
from src.data_ingestion import get_price_history

logger = get_logger(__name__)


def compute_sentiment_exhaustion_top(
    ticker: str,
    sentiment_score: float,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Detects contrarian sentiment exhaustion tops:
    When sentiment is euphoric (> 0.70) but price action is stagnating or stalling below recent highs.
    """
    if df is None or df.empty:
        try:
            df = get_price_history(ticker, period="3mo", use_cache=True)
        except Exception:
            df = pd.DataFrame()

    if df.empty or len(df) < 10:
        return {
            "ticker": ticker,
            "status": "INSUFFICIENT_DATA",
            "is_exhaustion_top": False,
            "divergence_score": 0.0,
            "warning": "Insufficient price series.",
        }

    closes = df["Close"].astype(float)
    recent_5d_return = float(closes.pct_change(5).iloc[-1] * 100.0)
    high_10d = float(df["High"].rolling(10).max().iloc[-1])
    current_price = float(closes.iloc[-1])
    distance_to_high_pct = ((high_10d - current_price) / high_10d) * 100.0

    # Euphoric sentiment with price lagging / failing to make new highs
    is_euphoric = sentiment_score >= 0.72
    is_price_lagging = recent_5d_return <= 0.5 or distance_to_high_pct >= 2.5

    is_exhaustion = is_euphoric and is_price_lagging
    divergence_score = round(
        max(0.0, (sentiment_score * 100.0) - (recent_5d_return * 5.0)), 1
    )
    divergence_score = min(100.0, divergence_score)

    return {
        "ticker": ticker,
        "status": "EVALUATION_COMPLETE",
        "sentiment_score": round(sentiment_score, 3),
        "recent_5d_return_pct": round(recent_5d_return, 2),
        "distance_to_10d_high_pct": round(distance_to_high_pct, 2),
        "is_exhaustion_top": is_exhaustion,
        "divergence_score": divergence_score,
        "thesis": (
            f"⚠️ Sentiment Exhaustion Warning: Euphoric sentiment ({sentiment_score:+.2f}) "
            f"uncoupled from flat 5-day return ({recent_5d_return:+.1f}%)."
            if is_exhaustion
            else "Price action aligns with sentiment flow."
        ),
    }


def compute_short_squeeze_probability(
    ticker: str,
    short_percent_of_float: Optional[float] = None,
    short_ratio_days: Optional[float] = None,
    sentiment_acceleration: float = 0.0,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Calculates Short Squeeze Probability Score (SSPS) [0, 100].
    Combines:
    - Short Interest % of Float (e.g. 15% - 40%)
    - Days to Cover (Short Ratio, e.g. 4 - 15 days)
    - Abnormal Volume Surge Ratio (Current Volume / 20-day SMA Volume)
    - Sentiment Acceleration (Velocity of positive mentions)
    """
    if df is None or df.empty:
        try:
            df = get_price_history(ticker, period="3mo", use_cache=True)
        except Exception:
            df = pd.DataFrame()

    vol_surge_ratio = 1.0
    if not df.empty and len(df) >= 20 and "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        sma_vol = vol.rolling(20).mean().iloc[-1]
        cur_vol = vol.iloc[-1]
        if sma_vol > 0:
            vol_surge_ratio = float(cur_vol / sma_vol)

    # Defaults for large cap / tech stocks if data not in proxy
    si_pct = short_percent_of_float if short_percent_of_float is not None else 3.5
    dtc = short_ratio_days if short_ratio_days is not None else 2.0

    # Sub-component scoring normalized to 0-100
    # 1. Short Interest Score (capped at 40% float)
    si_score = min(max(si_pct / 30.0 * 100.0, 0.0), 100.0)
    # 2. Days to Cover Score (capped at 8 days)
    dtc_score = min(max(dtc / 8.0 * 100.0, 0.0), 100.0)
    # 3. Volume Surge Score (capped at 3.0x normal volume)
    vol_score = min(max((vol_surge_ratio - 0.8) / 2.2 * 100.0, 0.0), 100.0)
    # 4. Sentiment Acceleration Score
    sent_score = min(max((sentiment_acceleration + 0.5) * 100.0, 0.0), 100.0)

    # Master Weighted SSPS Formula:
    # 35% Short Interest + 25% Days to Cover + 25% Volume Surge + 15% Sentiment
    ssps = 0.35 * si_score + 0.25 * dtc_score + 0.25 * vol_score + 0.15 * sent_score
    ssps = round(float(np.clip(ssps, 0.0, 100.0)), 1)

    if ssps >= 75.0:
        regime = "HIGH_SQUEEZE_POTENTIAL"
        recommendation = "BULLISH_GAMMA_SQUEEZE_ALERT"
    elif ssps >= 50.0:
        regime = "MODERATE_SQUEEZE_WATCH"
        recommendation = "MONITOR_ACCELERATION"
    else:
        regime = "LOW_SQUEEZE_RISK"
        recommendation = "STANDARD_FLOW"

    return {
        "ticker": ticker,
        "ssps_score": ssps,
        "squeeze_regime": regime,
        "recommendation": recommendation,
        "short_interest_pct": round(si_pct, 2),
        "days_to_cover": round(dtc, 1),
        "volume_surge_ratio": round(vol_surge_ratio, 2),
        "sentiment_acceleration": round(sentiment_acceleration, 3),
        "breakdown": {
            "short_interest_subscore": round(si_score, 1),
            "days_to_cover_subscore": round(dtc_score, 1),
            "volume_surge_subscore": round(vol_score, 1),
            "sentiment_velocity_subscore": round(sent_score, 1),
        },
    }
