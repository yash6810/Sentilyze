"""
Institutional Execution Engine & Smart Order Router (SOR) for Sentilyze.

Mechanics & Academic Grounding:
1. Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions:
   Calculates optimal trading trajectory balancing permanent/temporary market impact against volatility risk.
2. Bouchaud et al. (2009) - Square-Root Market Impact Law:
   Impact = Y * sigma * sqrt(Q / ADV).
3. Vectorized Intraday VWAP & Anti-Gaming Randomized TWAP.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def compute_almgren_chriss_trajectory(
    total_shares: float,
    total_time_hours: float = 6.5,
    n_intervals: int = 13,
    risk_aversion: float = 1e-6,
    daily_vol: float = 0.02,
    temp_impact: float = 2.5e-7,
    perm_impact: float = 2.5e-8,
) -> Dict[str, Any]:
    """
    Computes closed-form Almgren-Chriss (2000) optimal liquidation/acquisition schedule.

    Trajectory:
    kappa ~ sqrt(lambda * sigma^2 / eta)
    n_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X
    """
    if total_shares <= 0 or n_intervals <= 1:
        return {
            "schedule": [total_shares],
            "tranches": [total_shares],
            "total_shares": total_shares,
            "half_life_hours": 0.0,
        }

    T = float(total_time_hours)
    N = int(n_intervals)
    dt = T / N
    X = float(total_shares)

    # Volatility per hour
    sigma_hourly = daily_vol / np.sqrt(6.5)
    eta = max(temp_impact, 1e-9)
    lam = max(risk_aversion, 1e-9)

    # Urgency parameter kappa
    kappa2 = (lam * (sigma_hourly**2)) / eta
    kappa = np.sqrt(max(kappa2, 1e-8))

    t_grid = np.linspace(0, T, N + 1)
    sinh_kappa_T = np.sinh(min(kappa * T, 50.0))

    if sinh_kappa_T <= 1e-9 or np.isnan(sinh_kappa_T):
        # Linear TWAP fallback if kappa approaches 0
        holdings = X * (1.0 - t_grid / T)
    else:
        holdings = X * np.sinh(np.clip(kappa * (T - t_grid), 0, 50.0)) / sinh_kappa_T

    # Tranches to execute per interval
    tranches = np.diff(holdings * -1.0)
    tranches = np.maximum(tranches, 0.0)

    # Expected Cost: E[Cost] = 0.5 * gamma * X^2 + eta * sum(tranche^2 / dt)
    perm_cost = 0.5 * perm_impact * (X**2)
    temp_cost = eta * np.sum((tranches**2) / dt)
    expected_cost = perm_cost + temp_cost

    half_life = np.log(2.0) / max(kappa, 1e-6)

    schedule_points = []
    for i in range(len(tranches)):
        schedule_points.append(
            {
                "interval": i + 1,
                "time_hour": round(float(t_grid[i + 1]), 2),
                "shares_to_execute": round(float(tranches[i]), 1),
                "remaining_shares": round(float(holdings[i + 1]), 1),
                "pct_of_order": round(float(tranches[i] / X * 100.0), 2),
            }
        )

    return {
        "model": "Almgren-Chriss Optimal Execution",
        "total_shares": X,
        "n_intervals": N,
        "urgency_kappa": round(float(kappa), 5),
        "half_life_hours": round(float(half_life), 2),
        "expected_impact_cost": round(float(expected_cost), 2),
        "schedule": schedule_points,
    }


def compute_vwap_schedule(
    total_shares: float,
    n_intervals: int = 13,
) -> Dict[str, Any]:
    """
    Computes intraday Volume-Weighted Average Price (VWAP) execution curve.
    Uses canonical U-shaped intraday volume profile:
    - High volume at Open (9:30-10:00)
    - Low volume at Lunch (12:00-13:30)
    - Surge at Close (15:30-16:00)
    """
    if total_shares <= 0 or n_intervals <= 1:
        return {"model": "VWAP", "total_shares": total_shares, "schedule": []}

    # Normalized U-curve weights across trading day
    x = np.linspace(-1.0, 1.0, n_intervals)
    # Quadratic U-shape + baseline
    weights = 0.4 * (x**2) + 0.6
    weights = weights / np.sum(weights)

    tranches = total_shares * weights

    schedule = []
    cum_shares = 0.0
    for i in range(n_intervals):
        sh = float(tranches[i])
        cum_shares += sh
        schedule.append(
            {
                "interval": i + 1,
                "shares_to_execute": round(sh, 1),
                "pct_of_order": round(float(weights[i] * 100.0), 2),
                "cumulative_executed": round(cum_shares, 1),
            }
        )

    return {
        "model": "Intraday Volume-Weighted (VWAP)",
        "total_shares": total_shares,
        "n_intervals": n_intervals,
        "schedule": schedule,
    }


def compute_twap_schedule(
    total_shares: float,
    n_intervals: int = 13,
    randomize_pct: float = 0.15,
) -> Dict[str, Any]:
    """
    Computes Time-Weighted Average Price (TWAP) with stealth anti-gaming randomization.
    Adds +/- randomize_pct noise to tranche sizes to avoid front-running by HFT predatory algorithms.
    """
    if total_shares <= 0 or n_intervals <= 1:
        return {"model": "TWAP", "total_shares": total_shares, "schedule": []}

    np.random.seed(42)
    base_share = total_shares / float(n_intervals)
    noise = np.random.uniform(-randomize_pct, randomize_pct, n_intervals)
    raw_tranches = base_share * (1.0 + noise)
    # Re-normalize to exact total shares
    tranches = raw_tranches * (total_shares / np.sum(raw_tranches))

    schedule = []
    cum_shares = 0.0
    for i in range(n_intervals):
        sh = float(tranches[i])
        cum_shares += sh
        schedule.append(
            {
                "interval": i + 1,
                "shares_to_execute": round(sh, 1),
                "pct_of_order": round(float(sh / total_shares * 100.0), 2),
                "cumulative_executed": round(cum_shares, 1),
            }
        )

    return {
        "model": "Anti-Gaming Stealth TWAP",
        "total_shares": total_shares,
        "n_intervals": n_intervals,
        "schedule": schedule,
    }


def compute_bouchaud_market_impact(
    shares: float,
    spot_price: float,
    avg_daily_volume: float,
    daily_volatility: float = 0.02,
    y_factor: float = 0.60,
) -> Dict[str, Any]:
    """
    Computes square-root market impact using Bouchaud et al. (2009) universal law:
    Impact = Y * sigma * sqrt(Q / ADV)
    """
    adv = max(avg_daily_volume, 1000.0)
    q = max(shares, 0.0)
    participation_rate = q / adv

    # Square root law
    rel_impact = y_factor * daily_volatility * np.sqrt(participation_rate)
    rel_impact = float(np.clip(rel_impact, 0.0001, 0.05))  # 1 bps to 500 bps

    dollar_impact = spot_price * rel_impact
    impact_bps = rel_impact * 10000.0

    return {
        "shares": shares,
        "spot_price": spot_price,
        "participation_rate_pct": round(participation_rate * 100.0, 4),
        "impact_bps": round(impact_bps, 2),
        "dollar_slippage": round(dollar_impact, 4),
        "effective_buy_price": round(spot_price + dollar_impact, 4),
        "effective_sell_price": round(spot_price - dollar_impact, 4),
    }


def route_smart_order(
    ticker: str,
    shares: float,
    spot_price: float,
    strategy: str = "VWAP",
    adv: Optional[float] = None,
    daily_vol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Smart Order Routing (SOR) Dispatcher.
    Selects between ALMGREN_CHRISS, VWAP, and STEALTH_TWAP based on order size and urgency.
    """
    effective_adv = adv if adv and adv > 0 else 5_000_000.0
    effective_vol = daily_vol if daily_vol and daily_vol > 0 else 0.02

    # 1. Market Impact Assessment
    impact = compute_bouchaud_market_impact(
        shares=shares,
        spot_price=spot_price,
        avg_daily_volume=effective_adv,
        daily_volatility=effective_vol,
    )

    strat_upper = strategy.upper()
    if "ALMGREN" in strat_upper or "OPTIMAL" in strat_upper:
        plan = compute_almgren_chriss_trajectory(
            total_shares=shares,
            daily_vol=effective_vol,
        )
    elif "TWAP" in strat_upper:
        plan = compute_twap_schedule(total_shares=shares)
    else:
        plan = compute_vwap_schedule(total_shares=shares)

    return {
        "ticker": ticker,
        "order_shares": shares,
        "spot_price": spot_price,
        "strategy_applied": plan["model"],
        "market_impact": impact,
        "execution_plan": plan,
    }
