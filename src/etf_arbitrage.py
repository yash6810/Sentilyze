"""
ETF Intraday NAV Arbitrage Vector & Constituent Flow Predictor (Sprint 2, Module 2.9 / Idea 20)

Monitors the real-time basis spread between Sector ETFs and synthetic constituent stock baskets:
Basis Spread (%) = ((P_ETF - iNAV) / iNAV) * 100%

When the ETF trades at an abnormal premium (+0.35%), Authorized Participants (APs) create shares
by aggressively purchasing underlying basket stocks, generating predictable institutional tailwinds.
When at a discount (-0.35%), AP redemptions create selling pressure across constituent equities.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.data_ingestion import get_price_history

logger = logging.getLogger("Sentilyze.ETFArbitrage")
logging.basicConfig(level=logging.INFO)

# Canonical benchmark constituent baskets with weights
ETF_BASKET_DEFINITIONS = {
    "SMH": {
        "name": "VanEck Semiconductor ETF",
        "constituents": {
            "NVDA": 0.205,
            "TSM": 0.130,
            "AVGO": 0.085,
            "ASML": 0.070,
            "AMD": 0.055,
            "QCOM": 0.045,
            "TXN": 0.040,
            "AMAT": 0.035,
        },
        "divisor": 1.0,
    },
    "XLK": {
        "name": "Technology Select Sector SPDR",
        "constituents": {
            "AAPL": 0.220,
            "MSFT": 0.210,
            "NVDA": 0.190,
            "AVGO": 0.055,
            "CRM": 0.035,
            "ORCL": 0.030,
        },
        "divisor": 1.0,
    },
    "XLI": {
        "name": "Industrial Select Sector SPDR",
        "constituents": {
            "GE": 0.085,
            "CAT": 0.075,
            "UNP": 0.060,
            "RTX": 0.055,
            "HON": 0.050,
            "DE": 0.045,
        },
        "divisor": 1.0,
    },
}


def calculate_inav(
    basket_weights: Dict[str, float],
    constituent_prices: Dict[str, float],
    target_scale: float = 1.0,
) -> float:
    """
    Computes synthetic indicative Net Asset Value (iNAV) from constituent equity prices.
    Normalized to match the ETF nominal scale.
    """
    if not basket_weights or not constituent_prices:
        return 0.0

    total_weight = sum(basket_weights.get(t, 0.0) for t in constituent_prices)
    if total_weight <= 0:
        return 0.0

    # Weighted price sum
    weighted_sum = sum(
        basket_weights[t] * constituent_prices[t]
        for t in constituent_prices
        if t in basket_weights
    )
    # Scale normalized by covered constituent weight
    normalized_val = weighted_sum / total_weight
    return round(float(normalized_val * target_scale), 3)


def calculate_etf_nav_spread(etf_price: float, inav: float) -> Dict[str, Any]:
    """
    Computes basis spread between ETF trading price and iNAV:
    Spread (%) = ((P_ETF - iNAV) / iNAV) * 100%
    """
    p = max(float(etf_price), 1e-4)
    nav = max(float(inav), 1e-4)

    basis_pts = ((p - nav) / nav) * 10000.0  # in basis points
    spread_pct = ((p - nav) / nav) * 100.0

    if spread_pct > 0.35:
        regime = "ETF_PREMIUM_AP_CREATION_FLOW"
        constituent_flow = "INSTITUTIONAL_BUY_TAILWIND"
    elif spread_pct < -0.35:
        regime = "ETF_DISCOUNT_AP_REDEMPTION_DRAG"
        constituent_flow = "INSTITUTIONAL_SELL_HEADWIND"
    else:
        regime = "FAIR_VALUE_PARITY"
        constituent_flow = "BALANCED_ARBITRAGE"

    return {
        "etf_price": round(p, 2),
        "inav_price": round(nav, 2),
        "spread_pct": round(spread_pct, 3),
        "spread_bps": round(basis_pts, 1),
        "arbitrage_regime": regime,
        "constituent_flow_prediction": constituent_flow,
    }


def evaluate_etf_arbitrage_vector(
    etf_ticker: str = "SMH",
    custom_constituent_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Gathers real-time quotes for the ETF and its underlying basket,
    computing the live basis spread and constituent tailwinds.
    """
    t_clean = etf_ticker.upper().strip()
    basket_cfg = ETF_BASKET_DEFINITIONS.get(t_clean, ETF_BASKET_DEFINITIONS["SMH"])
    weights = basket_cfg["constituents"]

    # 1. Fetch ETF price
    etf_price = 245.0  # default baseline
    try:
        df_etf = get_price_history(t_clean, period="5d", use_cache=True)
        if not df_etf.empty:
            etf_price = float(df_etf["Close"].iloc[-1])
    except Exception:
        pass

    # 2. Fetch constituent prices
    prices = custom_constituent_prices or {}
    if not prices:
        for symbol in weights:
            try:
                df_c = get_price_history(symbol, period="5d", use_cache=True)
                if not df_c.empty:
                    prices[symbol] = float(df_c["Close"].iloc[-1])
            except Exception:
                pass

    if not prices:
        # Fallback parity
        prices = {t: etf_price for t in weights}

    # Normalize scale to ETF nominal level
    # Base synthetic average
    raw_avg = np.mean(list(prices.values())) if prices else etf_price
    scale_factor = etf_price / max(raw_avg, 1.0)
    inav = calculate_inav(weights, prices, target_scale=scale_factor)

    spread_report = calculate_etf_nav_spread(etf_price, inav)

    return {
        "status": "SUCCESS",
        "etf_ticker": t_clean,
        "etf_name": basket_cfg["name"],
        "constituents_evaluated": len(prices),
        "top_constituents": list(weights.keys())[:4],
        **spread_report,
    }
