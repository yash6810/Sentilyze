"""
Institutional Market Microstructure Engine (Sprint 1, Modules 1.5 & 1.6 / Ideas 12, 13, 19)

Implements:
1. Stoikov Micro-Price (Sasha Stoikov 2018):
   High-frequency fair-price estimator adjusting standard mid-price by order book volume imbalance.
2. Bulk Volume Classification (BVC) Flow Toxicity (Easley, Lopez de Prado & O'Hara 2012):
   Probabilistic volume decomposition into buyer/seller initiated trades via standard normal CDF.
3. Time-of-Day Volume Normalizer (Idea 19):
   Fits empirical U-shaped intraday volume curves (9:30 AM to 4:00 PM EST) to eliminate false morning breakout signals.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, time
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_ingestion import get_price_history

logger = logging.getLogger("Sentilyze.MicrostructureEngine")
logging.basicConfig(level=logging.INFO)


def calculate_stoikov_micro_price(
    bid_price: float,
    ask_price: float,
    bid_size: float,
    ask_size: float,
    alpha: float = 1.0,
) -> float:
    """
    Computes Stoikov Micro-Price (Stoikov 2018):
    P_micro = Mid + (Imbalance * Spread / 2) * alpha
    where Imbalance = (Q_b - Q_a) / (Q_b + Q_a)
    """
    if bid_price <= 0 or ask_price <= 0:
        return max(bid_price, ask_price)
    if bid_price >= ask_price:
        return (bid_price + ask_price) / 2.0

    spread = ask_price - bid_price
    mid_price = (bid_price + ask_price) / 2.0
    total_depth = bid_size + ask_size

    if total_depth <= 0:
        return round(mid_price, 4)

    imbalance = (bid_size - ask_size) / total_depth
    # Bound adjustment within the bid-ask corridor
    adjustment = imbalance * (spread / 2.0) * float(np.clip(alpha, 0.1, 1.5))
    micro_price = mid_price + adjustment
    return round(float(np.clip(micro_price, bid_price, ask_price)), 4)


def calculate_bar_stoikov_micro_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bar-level approximation of Stoikov Micro-Price from OHLCV bars.
    Uses Close relative to High-Low midpoint and Bar Imbalance.
    """
    if df.empty or not all(c in df.columns for c in ["High", "Low", "Close", "Volume"]):
        return df

    out = df.copy()
    hl_mid = (out["High"] + out["Low"]) / 2.0
    bar_range = (out["High"] - out["Low"]).replace(0, 1e-6)

    # Intrabar location ratio in [-1, +1]
    intrabar_loc = 2.0 * ((out["Close"] - out["Low"]) / bar_range) - 1.0

    # Synthetic spread proxy based on daily/intraday range
    spread_proxy = bar_range * 0.10

    # Stoikov micro-price estimate
    out["stoikov_micro_price"] = out["Close"] + (intrabar_loc * (spread_proxy / 2.0))
    out["micro_price_delta_pct"] = (
        (out["stoikov_micro_price"] - out["Close"]) / out["Close"]
    ) * 100.0
    return out


def calculate_bvc_flow_toxicity(
    df: pd.DataFrame,
    vol_col: str = "Volume",
    price_col: str = "Close",
    window: int = 20,
) -> pd.DataFrame:
    """
    Bulk Volume Classification (BVC) Flow Toxicity (Easley, Lopez de Prado & O'Hara 2012).
    Decomposes bar volume into Buyer-initiated V_B and Seller-initiated V_S:
    V_B = V * Phi( Delta P / sigma_delta_p )
    V_S = V - V_B
    """
    if df.empty or vol_col not in df.columns or price_col not in df.columns:
        return df

    out = df.copy()
    delta_p = out[price_col].diff().fillna(0.0)
    rolling_std = (
        delta_p.rolling(window=window, min_periods=5).std().replace(0, 1e-6).bfill()
    )

    # Standardized return z-score
    z = (delta_p / rolling_std).clip(-4.0, 4.0)

    # Normal CDF buy probability
    p_buy = pd.Series(norm.cdf(z.values), index=out.index)
    v_total = out[vol_col].clip(lower=0)

    out["bvc_buy_volume"] = v_total * p_buy
    out["bvc_sell_volume"] = v_total * (1.0 - p_buy)
    out["bvc_volume_delta"] = out["bvc_buy_volume"] - out["bvc_sell_volume"]
    out["bvc_cumulative_delta"] = out["bvc_volume_delta"].cumsum()

    # Toxicity index: proportion of absolute imbalance over total volume
    v_nonzero = v_total.replace(0, 1e-6)
    out["bvc_toxicity"] = (out["bvc_volume_delta"].abs() / v_nonzero).clip(0.0, 1.0)
    out["bvc_buyer_ratio"] = (out["bvc_buy_volume"] / v_nonzero).clip(0.0, 1.0)

    return out


class TimeOfDayVolumeNormalizer:
    """
    Fits and applies empirical U-shaped intraday volume curves (Idea 19).
    Eliminates false morning breakout spikes by scaling raw volume relative
    to its expected time-of-day baseline.
    """

    # Typical U-curve weightings for 13 half-hour buckets (9:30 AM to 4:00 PM EST)
    DEFAULT_U_CURVE = {
        time(9, 30): 2.45,  # Morning open rush (high)
        time(10, 0): 1.65,
        time(10, 30): 1.20,
        time(11, 0): 0.90,
        time(11, 30): 0.70,  # Lunch lull (low)
        time(12, 0): 0.60,
        time(12, 30): 0.58,
        time(13, 0): 0.65,
        time(13, 30): 0.75,
        time(14, 0): 0.95,
        time(14, 30): 1.15,
        time(15, 0): 1.50,
        time(15, 30): 2.30,  # Market-On-Close rush (high)
    }

    def __init__(self, custom_u_curve: Optional[Dict[time, float]] = None):
        self.u_curve = custom_u_curve or self.DEFAULT_U_CURVE

    def get_time_multiplier(self, target_time: time) -> float:
        """Finds closest time bucket multiplier."""
        # Find closest match in dictionary
        minutes_target = target_time.hour * 60 + target_time.minute
        closest_mult = 1.0
        min_diff = float("inf")

        for bucket_time, mult in self.u_curve.items():
            diff = abs((bucket_time.hour * 60 + bucket_time.minute) - minutes_target)
            if diff < min_diff:
                min_diff = diff
                closest_mult = mult

        return closest_mult

    def normalize_volume(
        self,
        raw_volume: float,
        timestamp: Optional[datetime] = None,
        average_bucket_volume: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Normalizes raw volume by expected time-of-day baseline.
        Returns Relative Volume (RVOL) free from diurnal U-curve distortion.
        """
        if timestamp is None:
            t_now = datetime.now().time()
        else:
            t_now = timestamp.time()

        mult = self.get_time_multiplier(t_now)
        expected_volume = max(average_bucket_volume * mult, 1.0)
        rvol = raw_volume / expected_volume

        is_breakout_volume = rvol >= 2.0
        is_subdued_volume = rvol <= 0.6

        return {
            "raw_volume": raw_volume,
            "u_curve_multiplier": round(mult, 2),
            "expected_diurnal_volume": round(expected_volume, 1),
            "normalized_rvol": round(rvol, 2),
            "is_breakout_confirmed": bool(is_breakout_volume),
            "is_subdued_volume": bool(is_subdued_volume),
        }


def evaluate_microstructure_telemetry(
    ticker: str, spot_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates comprehensive microstructure signals for an asset:
    - Bar Stoikov Micro-price vs Close
    - Bulk Volume Classification (BVC) toxicity
    - Diurnal Time-of-Day Relative Volume
    """
    try:
        df = get_price_history(ticker, period="5d", interval="5m", use_cache=True)
    except Exception:
        df = pd.DataFrame()

    if df.empty or len(df) < 5:
        # Fallback to daily data if intraday is unavailable
        try:
            df = get_price_history(ticker, period="3mo", use_cache=True)
        except Exception:
            df = pd.DataFrame()

    if df.empty or len(df) < 5:
        return {
            "status": "INSUFFICIENT_DATA",
            "stoikov_micro_price": spot_price or 100.0,
            "bvc_toxicity": 0.35,
            "bvc_buyer_ratio": 0.50,
            "microstructure_verdict": "NEUTRAL_FLOW",
        }

    # 1. Stoikov Micro-price
    df_stoikov = calculate_bar_stoikov_micro_price(df)
    latest_bar = df_stoikov.iloc[-1]
    micro_price = float(latest_bar.get("stoikov_micro_price", spot_price or 100.0))
    micro_delta_pct = float(latest_bar.get("micro_price_delta_pct", 0.0))

    # 2. BVC Toxicity
    df_bvc = calculate_bvc_flow_toxicity(df_stoikov)
    latest_bvc = df_bvc.iloc[-1]
    bvc_tox = float(latest_bvc.get("bvc_toxicity", 0.35))
    buyer_ratio = float(latest_bvc.get("bvc_buyer_ratio", 0.50))
    cum_delta = float(latest_bvc.get("bvc_cumulative_delta", 0.0))

    # 3. Time-of-Day Normalizer
    normalizer = TimeOfDayVolumeNormalizer()
    raw_vol = float(latest_bar.get("Volume", 100000.0))
    avg_vol = float(df["Volume"].tail(30).mean())
    vol_norm = normalizer.normalize_volume(raw_vol, average_bucket_volume=avg_vol)

    # Formulate verdict
    if bvc_tox > 0.65 and buyer_ratio < 0.35:
        verdict = "TOXIC_SELLING_FLOW"
    elif bvc_tox > 0.65 and buyer_ratio > 0.65:
        verdict = "AGGRESSIVE_BUYER_ABSORPTION"
    elif micro_delta_pct > 0.20:
        verdict = "MICRO_PRICE_BULLISH_SKEW"
    elif micro_delta_pct < -0.20:
        verdict = "MICRO_PRICE_BEARISH_SKEW"
    else:
        verdict = "BALANCED_ORDER_FLOW"

    # Kyle's Lambda Price Impact
    kyle_meta = calculate_kyles_lambda(df)

    return {
        "status": "SUCCESS",
        "ticker": ticker,
        "stoikov_micro_price": round(micro_price, 2),
        "micro_price_delta_pct": round(micro_delta_pct, 3),
        "bvc_toxicity": round(bvc_tox, 3),
        "bvc_buyer_ratio": round(buyer_ratio, 3),
        "bvc_cumulative_delta": round(cum_delta, 1),
        "rvol_normalized": vol_norm.get("normalized_rvol", 1.0),
        "is_volume_breakout": vol_norm.get("is_breakout_confirmed", False),
        "kyles_lambda": kyle_meta.get("kyles_lambda", 0.0),
        "kyles_depth_dollars": kyle_meta.get("implied_market_depth_dollars", 0.0),
        "microstructure_verdict": verdict,
    }


def calculate_kyles_lambda(
    df: pd.DataFrame,
    price_col: str = "Close",
    vol_col: str = "Volume",
) -> Dict[str, Any]:
    """
    Kyle's Lambda (Kyle 1985) Price Impact Estimator (Sprint 2, Module 2.5 / Idea 14).
    Regresses price changes against signed square-root dollar volume:
    Delta P_t = lambda * (Sign_t * sqrt(Dollar_Vol_t)) + epsilon_t
    lambda = Cov(Delta P, Q) / Var(Q)
    """
    if (
        df.empty
        or len(df) < 10
        or price_col not in df.columns
        or vol_col not in df.columns
    ):
        return {"kyles_lambda": 0.0, "implied_market_depth_dollars": 1e7}

    delta_p = df[price_col].diff().fillna(0.0)
    dollar_vol = (df[price_col] * df[vol_col]).clip(lower=1.0)
    signed_flow = np.sign(delta_p) * np.sqrt(dollar_vol)

    var_q = np.var(signed_flow)
    if var_q <= 1e-12:
        return {"kyles_lambda": 0.0, "implied_market_depth_dollars": 1e7}

    cov_pq = np.cov(delta_p, signed_flow)[0, 1]
    lambda_kyle = float(max(cov_pq / var_q, 0.0))

    # Implied depth: dollar volume required to move price by $1.00
    implied_depth = float((1.0 / max(lambda_kyle, 1e-8)) ** 2)

    return {
        "kyles_lambda": round(lambda_kyle, 6),
        "implied_market_depth_dollars": round(implied_depth, 2),
        "liquidity_classification": (
            "DEEP_LIQUIDITY" if lambda_kyle < 0.0001 else "THIN_SLIPPAGE_RISK"
        ),
    }


def detect_iceberg_order_tape(
    displayed_depth: float,
    executed_tape_volume: float,
    price_level: float,
    min_multiplier: float = 3.0,
    is_bid: bool = True,
) -> Dict[str, Any]:
    """
    Iceberg Order Tape Spotter (Sprint 2, Module 2.3 / Idea 16).
    Detects hidden institutional orders when executed tape prints at a price level
    greatly exceed the visible displayed level depth.
    """
    displayed = max(float(displayed_depth), 1.0)
    executed = max(float(executed_tape_volume), 0.0)
    multiplier = executed / displayed

    is_iceberg = bool(multiplier >= min_multiplier and executed >= (displayed * 2.0))
    side = "BID_ACCUMULATION" if is_bid else "ASK_DISTRIBUTION"

    return {
        "price_level": round(price_level, 2),
        "displayed_depth": displayed,
        "executed_tape_volume": executed,
        "hidden_multiplier": round(multiplier, 2),
        "is_iceberg_detected": is_iceberg,
        "side": side,
        "institutional_signal": f"ICEBERG_{side}" if is_iceberg else "NORMAL_DEPTH",
    }


def calculate_order_book_resilience(
    initial_depth: float,
    depth_post_sweep: float,
    depth_recovered: float,
    recovery_time_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Order Book Resilience & Replenishment Monitor (Sprint 2, Module 2.4 / Idea 15).
    Measures the speed and ratio of depth restoration following aggressive liquidity sweeps.
    """
    init_d = max(float(initial_depth), 1.0)
    recovered_d = max(float(depth_recovered), 0.0)

    replenishment_ratio = recovered_d / init_d
    resilience_speed = replenishment_ratio / max(recovery_time_seconds, 0.1)

    is_fake_liquidity = bool(replenishment_ratio < 0.35 and recovery_time_seconds > 5.0)
    is_genuine_institutional = bool(
        replenishment_ratio >= 0.80 and recovery_time_seconds <= 3.0
    )

    verdict = "NORMAL_LIQUIDITY_RECOVERY"
    if is_fake_liquidity:
        verdict = "PHANTOM_LIQUIDITY_FAKE_WALL"
    elif is_genuine_institutional:
        verdict = "GENUINE_INSTITUTIONAL_ABSORPTION"

    return {
        "replenishment_ratio": round(replenishment_ratio, 3),
        "recovery_time_seconds": round(recovery_time_seconds, 1),
        "resilience_velocity": round(resilience_speed, 3),
        "is_fake_liquidity_wall": is_fake_liquidity,
        "is_genuine_replenishment": is_genuine_institutional,
        "resilience_verdict": verdict,
    }


def calculate_l2_depth_weighted_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    decay_gamma: float = 0.5,
) -> Dict[str, Any]:
    """
    Level-2 Multi-Level Order Book Imbalance (Sprint 2, Module 2.11 / Idea 11).
    Computes depth-weighted imbalance across top K levels:
    I_L2 = sum(w_k * (Q_b^k - Q_a^k)) / sum(w_k * (Q_b^k + Q_a^k))
    where w_k = exp(-gamma * (k - 1))
    """
    if not bids or not asks:
        return {
            "l2_imbalance": 0.0,
            "imbalance_regime": "NEUTRAL",
            "levels_evaluated": 0,
        }

    k_max = min(len(bids), len(asks), 10)
    weighted_bid_q = 0.0
    weighted_ask_q = 0.0

    for k in range(k_max):
        w_k = np.exp(-decay_gamma * k)
        q_b = float(bids[k][1])
        q_a = float(asks[k][1])
        weighted_bid_q += w_k * q_b
        weighted_ask_q += w_k * q_a

    total_weighted = weighted_bid_q + weighted_ask_q
    if total_weighted <= 0:
        return {
            "l2_imbalance": 0.0,
            "imbalance_regime": "NEUTRAL",
            "levels_evaluated": k_max,
        }

    imbalance = (weighted_bid_q - weighted_ask_q) / total_weighted
    imbalance = float(np.clip(imbalance, -1.0, 1.0))

    if imbalance > 0.35:
        regime = "STRONG_BID_SUPPORT_STACKING"
    elif imbalance < -0.35:
        regime = "STRONG_ASK_OVERHANG_DISTRIBUTION"
    else:
        regime = "BALANCED_L2_ORDER_BOOK"

    return {
        "l2_imbalance": round(imbalance, 3),
        "weighted_bid_depth": round(weighted_bid_q, 1),
        "weighted_ask_depth": round(weighted_ask_q, 1),
        "levels_evaluated": k_max,
        "imbalance_regime": regime,
    }
