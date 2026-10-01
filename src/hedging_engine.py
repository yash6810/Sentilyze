"""
Institutional Hedging Engine & Convex Tail-Risk Derivatives Modeler
(Sprint 4, Modules 4.7 & 4.8 / Ideas 37 & 34)

Features:
1. Synthetic Down-and-Out Floor (Dynamic Delta-Gamma Futures Replication - Idea 37)
2. VIX Call Ratio Backspread (Convex Crash Tail-Risk Hedging - Idea 34)
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_synthetic_floor_hedge(
    portfolio_equity: float,
    spot_index: float,
    floor_pct: float = 0.95,
    volatility: float = 0.18,
    time_to_expiry_years: float = 0.25,
    barrier_pct: Optional[float] = 0.85,
    risk_free_rate: float = 0.045,
) -> Dict[str, Any]:
    """
    Synthetically replicates a Down-and-Out Put Floor protection using dynamic delta-gamma hedging (Idea 37).
    Guarantees portfolio equity does not fall below floor level F = Equity * floor_pct,
    while deactivating at barrier B to eliminate cost bleed during extreme recoveries.
    """
    S = float(spot_index)
    E = float(portfolio_equity)
    r = float(risk_free_rate)
    sigma = max(float(volatility), 0.05)
    T = max(float(time_to_expiry_years), 1.0 / 252.0)

    # Strike price corresponding to floor equity
    K = S * float(floor_pct)
    barrier = S * float(barrier_pct) if barrier_pct is not None else 0.0

    # Barrier active check
    is_barrier_breached = S <= barrier if barrier_pct is not None else False

    if is_barrier_breached:
        return {
            "status": "BARRIER_KNOCKED_OUT",
            "hedge_active": False,
            "hedge_ratio": 0.0,
            "hedge_short_dollars": 0.0,
            "floor_level_dollars": round(E * floor_pct, 2),
            "barrier_level": round(barrier, 2),
            "delta": 0.0,
            "gamma": 0.0,
        }

    # Standard Black-Scholes Put Delta and Gamma
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_delta = float(stats.norm.cdf(d1) - 1.0)  # in [-1, 0]
    gamma = float(stats.norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    # Down-and-Out adjustment factor (if barrier specified)
    if barrier > 0:
        mu = (r - 0.5 * sigma**2) / (sigma**2)
        barrier_factor = max(0.0, 1.0 - (barrier / S) ** (2.0 * mu))
        put_delta = put_delta * barrier_factor
        gamma = gamma * barrier_factor

    # Hedge Ratio: fraction of portfolio equity to short via index futures
    hedge_ratio = abs(put_delta)
    hedge_dollars = round(E * hedge_ratio, 2)
    contracts_equiv = round(hedge_dollars / max(S, 1.0), 2)

    return {
        "status": "ACTIVE",
        "hedge_active": True,
        "floor_level_dollars": round(E * floor_pct, 2),
        "floor_strike": round(K, 2),
        "barrier_strike": round(barrier, 2),
        "put_delta": round(put_delta, 4),
        "gamma": round(gamma, 6),
        "recommended_hedge_ratio": round(hedge_ratio, 4),
        "hedge_short_dollars": hedge_dollars,
        "index_equivalent_shares": contracts_equiv,
        "cash_cushion_required": round(E * (1.0 - floor_pct), 2),
    }


def calculate_vix_call_ratio_backspread(
    vix_spot: float = 16.0,
    strike_short: float = 18.0,
    strike_long: float = 24.0,
    ratio: int = 2,
    premium_short: float = 2.40,
    premium_long: float = 0.90,
    contracts_short: int = 10,
) -> Dict[str, Any]:
    """
    Constructs an institutional VIX 1xN Call Ratio Backspread (Idea 34 / Sprint 4.8).
    Sell 1 NTM Call (Strike K1) to finance buying N OTM Calls (Strike K2 > K1).

    Characteristics:
    - Zero or positive net credit upon entry (no theta bleed during quiet bull markets)
    - Controlled, bounded maximum loss if VIX stalls exactly at K2
    - Explosive convex non-linear payout if VIX spikes to 30, 40, or 60+ in a market crash.
    """
    K1 = float(strike_short)
    K2 = float(strike_long)
    N_ratio = max(int(ratio), 2)
    P1 = float(premium_short)
    P2 = float(premium_long)
    contracts = max(int(contracts_short), 1)

    # Net cost per 1xN spread unit
    # Sell 1 @ P1, Buy N @ P2
    net_cost_per_spread = (N_ratio * P2) - P1
    is_credit = net_cost_per_spread <= 0.0
    net_entry_cash_flow = -net_cost_per_spread * contracts * 100.0

    # Max Loss occurs when VIX expires exactly at K2:
    # Short call loss: -(K2 - K1), Long calls expire worthless
    # Total loss per spread: (K2 - K1) + net_cost_per_spread
    max_loss_per_spread = (K2 - K1) + net_cost_per_spread
    max_loss_total = round(max_loss_per_spread * contracts * 100.0, 2)

    # Upper Breakeven Point: K2 + max_loss_per_spread / (N - 1)
    upper_breakeven = round(K2 + (max_loss_per_spread / float(N_ratio - 1)), 2)

    # PnL Grid across VIX scenarios
    vix_scenarios = [14.0, 16.0, 18.0, 20.0, 24.0, 30.0, 40.0, 55.0, 75.0]
    pnl_grid = {}

    for v in vix_scenarios:
        # Payoff of short call
        payoff_short = -max(v - K1, 0.0)
        # Payoff of N long calls
        payoff_long = N_ratio * max(v - K2, 0.0)
        net_payoff = (
            (payoff_short + payoff_long - net_cost_per_spread) * contracts * 100.0
        )
        pnl_grid[f"VIX_{int(v)}"] = round(net_payoff, 2)

    return {
        "status": "SUCCESS",
        "structure": f"Sell 1x {K1:.0f}C / Buy {N_ratio}x {K2:.0f}C",
        "contracts_short": contracts,
        "contracts_long": contracts * N_ratio,
        "net_cost_per_spread": round(net_cost_per_spread, 2),
        "is_net_credit_entry": is_credit,
        "net_entry_cash_flow_dollars": round(net_entry_cash_flow, 2),
        "max_risk_dollars": max_loss_total,
        "upper_breakeven_vix": upper_breakeven,
        "pnl_scenarios_dollars": pnl_grid,
        "crash_protection_vix40_dollars": pnl_grid.get("VIX_40", 0.0),
        "super_crash_protection_vix75_dollars": pnl_grid.get("VIX_75", 0.0),
    }
