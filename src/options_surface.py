"""
3D Implied Volatility Surface & Multi-Leg Options Strategy Desk for Sentilyze.
Computes 3D Volatility Surfaces (Strike x Expiration x Implied Volatility)
and calculates payoff structures for multi-leg option strategies (Bull Call Spreads, Iron Condors, Straddles).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger
from src.realtime_tracker import fetch_live_quote

logger = get_logger(__name__)


def generate_volatility_surface_mesh(
    ticker: str, spot_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Constructs a 3D Implied Volatility Surface across strike prices and expiration dates.
    Models the institutional volatility smile/skew and term structure.
    """
    if spot_price is None or spot_price <= 0.0:
        quote = fetch_live_quote(ticker)
        spot_price = float(quote.get("price", 100.0))

    # Expirations in Days to Expiry (DTE)
    dtes = np.array([7, 14, 30, 45, 60, 90, 180])

    # Strikes from -20% to +20% Moneyness
    strike_multipliers = np.linspace(0.80, 1.20, 15)
    strikes = np.round(spot_price * strike_multipliers, 2)

    # Generate 2D Grid
    K_grid, T_grid = np.meshgrid(strikes, dtes)

    # Base ATM IV ~ 35% with skew (higher IV for OTM Puts) and term structure
    moneyness = K_grid / spot_price
    atm_iv = 0.35

    # Skew formula: Higher IV for lower strikes (crashophobia), slight uptick for far OTM calls
    skew_component = 0.25 * (1.0 - moneyness) + 0.15 * (moneyness - 1.0) ** 2

    # Term structure: IV rises slightly for longer tenors (mean-reverting uncertainty)
    term_component = 0.05 * np.log(T_grid / 30.0 + 1.0)

    # 3D Implied Volatility Matrix
    iv_matrix = np.clip(atm_iv + skew_component + term_component, 0.15, 0.95) * 100.0

    return {
        "ticker": ticker,
        "spot_price": spot_price,
        "strikes": strikes.tolist(),
        "dtes": dtes.tolist(),
        "iv_matrix": iv_matrix.tolist(),
        "atm_iv_pct": round(float(atm_iv * 100.0), 1),
    }


def calculate_multileg_payoff(
    strategy_type: str,
    spot_price: float,
    underlying_range_pct: float = 0.20,
) -> Dict[str, Any]:
    """
    Calculates profit and loss (P&L) curves at expiration for institutional multi-leg option structures.
    """
    p_min = spot_price * (1.0 - underlying_range_pct)
    p_max = spot_price * (1.0 + underlying_range_pct)
    price_range = np.linspace(p_min, p_max, 50)

    if strategy_type == "BULL_CALL_SPREAD":
        # Buy ATM Call, Sell OTM Call (+5%)
        k1 = round(spot_price, 2)
        k2 = round(spot_price * 1.05, 2)
        cost_k1 = round(spot_price * 0.04, 2)
        credit_k2 = round(spot_price * 0.018, 2)
        net_debit = cost_k1 - credit_k2
        max_profit = (k2 - k1) - net_debit
        max_loss = net_debit

        payoff = (
            np.maximum(price_range - k1, 0)
            - np.maximum(price_range - k2, 0)
            - net_debit
        )
        legs = [
            {"leg": "Long Call", "strike": k1, "type": "BUY", "premium": cost_k1},
            {"leg": "Short Call", "strike": k2, "type": "SELL", "premium": credit_k2},
        ]
        desc = f"Bullish defined-risk spread (Long ${k1:,.2f} Call / Short ${k2:,.2f} Call)."

    elif strategy_type == "IRON_CONDOR":
        # OTM Put Spread (k1, k2) + OTM Call Spread (k3, k4)
        k1 = round(spot_price * 0.90, 2)
        k2 = round(spot_price * 0.95, 2)
        k3 = round(spot_price * 1.05, 2)
        k4 = round(spot_price * 1.10, 2)

        net_credit = round(spot_price * 0.025, 2)
        wing_width = k2 - k1
        max_profit = net_credit
        max_loss = wing_width - net_credit

        put_payoff = -(
            np.maximum(k2 - price_range, 0) - np.maximum(k1 - price_range, 0)
        )
        call_payoff = -(
            np.maximum(price_range - k3, 0) - np.maximum(price_range - k4, 0)
        )
        payoff = put_payoff + call_payoff + net_credit

        legs = [
            {"leg": "Long Put", "strike": k1, "type": "BUY", "premium": 1.20},
            {"leg": "Short Put", "strike": k2, "type": "SELL", "premium": 2.40},
            {"leg": "Short Call", "strike": k3, "type": "SELL", "premium": 2.30},
            {"leg": "Long Call", "strike": k4, "type": "BUY", "premium": 1.00},
        ]
        desc = f"Market-neutral range bound structure collecting ${net_credit:.2f} credit between ${k2:,.2f} and ${k3:,.2f}."

    elif strategy_type == "LONG_STRADDLE":
        # Long ATM Call + Long ATM Put
        k_atm = round(spot_price, 2)
        cost_call = round(spot_price * 0.038, 2)
        cost_put = round(spot_price * 0.035, 2)
        net_debit = cost_call + cost_put
        max_loss = net_debit
        max_profit = float("inf")

        payoff = (
            np.maximum(price_range - k_atm, 0)
            + np.maximum(k_atm - price_range, 0)
            - net_debit
        )
        legs = [
            {
                "leg": "Long ATM Call",
                "strike": k_atm,
                "type": "BUY",
                "premium": cost_call,
            },
            {
                "leg": "Long ATM Put",
                "strike": k_atm,
                "type": "BUY",
                "premium": cost_put,
            },
        ]
        desc = f"Volatility breakout play expecting extreme move beyond ±{(net_debit/spot_price)*100:.1f}%."

    else:
        # Default Bear Put Spread
        k1 = round(spot_price * 0.95, 2)
        k2 = round(spot_price, 2)
        cost_k2 = round(spot_price * 0.035, 2)
        credit_k1 = round(spot_price * 0.015, 2)
        net_debit = cost_k2 - credit_k1
        max_profit = (k2 - k1) - net_debit
        max_loss = net_debit

        payoff = (
            np.maximum(k2 - price_range, 0)
            - np.maximum(k1 - price_range, 0)
            - net_debit
        )
        legs = [
            {"leg": "Long Put", "strike": k2, "type": "BUY", "premium": cost_k2},
            {"leg": "Short Put", "strike": k1, "type": "SELL", "premium": credit_k1},
        ]
        desc = (
            f"Bearish defined-risk spread (Long ${k2:,.2f} Put / Short ${k1:,.2f} Put)."
        )

    return {
        "strategy_type": strategy_type,
        "spot_price": spot_price,
        "description": desc,
        "max_profit": max_profit if max_profit != float("inf") else "Unlimited",
        "max_loss": max_loss,
        "risk_reward_ratio": (
            round(float(max_profit) / max(float(max_loss), 0.01), 2)
            if max_profit != "Unlimited"
            else 999.0
        ),
        "legs": legs,
        "price_range": price_range.tolist(),
        "payoff_curve": payoff.tolist(),
    }
