"""
Institutional ARAR Microstructure & Iceberg Absorption Engine for Sentilyze.
Formulated from institutional order book dynamics (Stoikov 2018, Easley-de Prado-O'Hara 2012):

1. The Wall: Order Book Imbalance (OBI) from resting depth / quote dynamics:
   OBI = (Bid_Vol - Ask_Vol) / (Bid_Vol + Ask_Vol)
2. The Attack: Cumulative Volume Delta (CVD) decomposing aggressive taker flow:
   CVD = Aggressive_Buys - Aggressive_Sells
3. The Trigger: Price Delta velocity (ΔPrice ≈ 0 indicates immovable absorption wall):
   ΔPrice = Price_now - Price_t-dt
4. The Confirmation: Volume-Synchronized Probability of Toxicity (VPIN):
   High VPIN confirms informed institutional flow.
5. ARAR Ratio (Aggressive-Passive Absorption Ratio):
   ARAR = (|CVD_sell| * OBI_bid) / (|ΔPrice| + ε)
   A spike in ARAR confirms aggressive sellers are exhausting themselves against an institutional iceberg wall.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import norm
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_order_book_imbalance(
    bid_volume: float,
    ask_volume: float,
) -> float:
    """
    Computes Level-1 Order Book Imbalance (OBI).
    Returns value in [-1.0, +1.0].
    +1.0 indicates massive institutional bid wall; -1.0 indicates massive ask overhang.
    """
    total_depth = float(bid_volume + ask_volume)
    if total_depth <= 0:
        return 0.0
    obi = (bid_volume - ask_volume) / total_depth
    return float(np.clip(obi, -1.0, 1.0))


def calculate_bar_cumulative_volume_delta(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Computes Cumulative Volume Delta (CVD) from intraday OHLCV bars using Bulk Volume Classification (BVC).
    Decomposes volume into aggressive buyers and aggressive sellers via normal CDF of return z-scores.
    """
    if df.empty or len(df) < 5 or not all(c in df.columns for c in ["Close", "Volume"]):
        out = df.copy() if not df.empty else pd.DataFrame()
        out["cvd"] = 0.0
        out["volume_delta"] = 0.0
        return out

    out = df.copy()
    close = out["Close"].values.astype(float)
    vol = out["Volume"].values.astype(float)

    delta_p = np.diff(close, prepend=close[0])
    rolling_std = (
        pd.Series(delta_p)
        .rolling(window=window, min_periods=3)
        .std()
        .fillna(1e-4)
        .values
    )
    rolling_std = np.maximum(rolling_std, 1e-6)

    # Standardized return z-score
    z = np.clip(delta_p / rolling_std, -4.0, 4.0)

    # Normal CDF gives buy volume fraction
    p_buy = norm.cdf(z)
    v_buy = vol * p_buy
    v_sell = vol * (1.0 - p_buy)

    vol_delta = v_buy - v_sell
    cvd = np.cumsum(vol_delta)

    out["buy_volume"] = v_buy
    out["sell_volume"] = v_sell
    out["volume_delta"] = vol_delta
    out["cvd"] = cvd
    return out


def compute_arar_absorption_ratio(
    cvd_sell: float,
    obi_bid: float,
    delta_price: float,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes the Aggressive-Passive Absorption Ratio (ARAR):
    ARAR = (|CVD_sell| * OBI_bid) / (|ΔPrice| + ε)

    Spikes when aggressive market sellers hit the bid (CVD_sell is large),
    the bid wall is thick (OBI_bid > 0), but the price displacement is near zero.
    """
    numerator = abs(float(cvd_sell)) * max(0.0, float(obi_bid))
    denominator = abs(float(delta_price)) + epsilon
    arar = numerator / denominator
    return float(arar)


def evaluate_arar_microstructure(
    arg1: Any,
    arg2: Any = None,
    vpin_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Comprehensive ARAR Microstructure Evaluation on 1m/5m intraday bars.
    Detects whether institutional iceberg walls are absorbing aggressive sell sweeps.
    Accepts either (df, threshold) or (ticker, df, threshold).
    """
    if isinstance(arg1, str) and isinstance(arg2, pd.DataFrame):
        ticker = arg1
        df_intraday = arg2
    elif isinstance(arg1, pd.DataFrame):
        df_intraday = arg1
        ticker = "UNKNOWN"
        if isinstance(arg2, (int, float)):
            vpin_threshold = float(arg2)
    else:
        df_intraday = pd.DataFrame()
        ticker = "UNKNOWN"

    if df_intraday.empty or len(df_intraday) < 10:
        return {
            "arar_score": 0.0,
            "obi": 0.0,
            "cvd": 0.0,
            "delta_price": 0.0,
            "is_absorption_active": False,
            "absorption_verdict": "INSUFFICIENT_DATA",
            "microstructure_signal": 0.0,
        }

    df_cvd = calculate_bar_cumulative_volume_delta(df_intraday)
    latest_bar = df_cvd.iloc[-1]
    prev_bar = df_cvd.iloc[-2]

    close = float(latest_bar["Close"])
    prev_close = float(prev_bar["Close"])
    high = float(latest_bar.get("High", close))
    low = float(latest_bar.get("Low", close))
    bar_range = max(1e-4, high - low)

    # 1. Level-1 OBI Proxy: Intrabar close position relative to midpoint
    # When close is high despite a sell-off bar, passive bids absorbed the dump
    intrabar_loc = 2.0 * ((close - low) / bar_range) - 1.0  # [-1, +1]
    obi_proxy = float(np.clip(intrabar_loc, -1.0, 1.0))

    # 2. Cumulative Volume Delta & Aggressive Selling
    vol_delta = float(latest_bar.get("volume_delta", 0.0))
    sell_vol = float(latest_bar.get("sell_volume", 0.0))
    cvd_latest = float(latest_bar.get("cvd", 0.0))

    # 3. Price Delta (Relative price movement)
    delta_p_pct = ((close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

    # 4. Compute ARAR
    # High selling volume into a holding price level creates a massive ARAR spike
    arar_score = compute_arar_absorption_ratio(
        cvd_sell=sell_vol,
        obi_bid=max(0.0, obi_proxy),
        delta_price=abs(delta_p_pct),
        epsilon=0.05,
    )

    # Normalize ARAR score to [0.0, 1.0] for model consumption
    arar_norm = float(np.tanh(arar_score / 500.0))

    # 5. Determine Absorption Long Reversal Trigger
    # Condition: Aggressive seller dump (vol_delta < 0), price closed in upper 40% of bar, OBI positive
    is_seller_dump = vol_delta < 0 or sell_vol > float(
        latest_bar.get("buy_volume", 0.0)
    )
    is_wall_holding = (close - low) >= (bar_range * 0.40)
    is_absorption_active = bool(
        is_seller_dump and is_wall_holding and arar_norm >= 0.40
    )

    # Formulate Microstructure Alpha Signal S_mu in [-1.0, +1.0]
    if is_absorption_active:
        micro_signal = float(np.clip(0.50 + 0.50 * arar_norm, 0.50, 1.0))
        verdict = "INSTITUTIONAL_ICEBERG_ABSORPTION_LONG"
    elif obi_proxy > 0.30 and vol_delta > 0:
        micro_signal = float(np.clip(obi_proxy * 0.70, 0.10, 0.70))
        verdict = "BULLISH_ORDER_FLOW"
    elif obi_proxy < -0.30 and vol_delta < 0:
        micro_signal = float(np.clip(obi_proxy * 0.70, -0.70, -0.10))
        verdict = "BEARISH_ORDER_OVERHANG"
    else:
        micro_signal = 0.0
        verdict = "NEUTRAL_FLOW"

    return {
        "arar_score": round(arar_score, 2),
        "arar_normalized": round(arar_norm, 3),
        "obi": round(obi_proxy, 3),
        "cvd": round(cvd_latest, 1),
        "delta_price_pct": round(delta_p_pct, 3),
        "is_absorption_active": is_absorption_active,
        "absorption_verdict": verdict,
        "microstructure_signal": round(micro_signal, 3),
    }
