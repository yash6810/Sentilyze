"""
Paper 18: Grossman-Zhou Optimal Drawdown-Constrained Strategy.

Source: Grossman & Zhou (1993) — "Optimal Investment Strategies for
Controlling Drawdowns", Mathematical Finance.
Complexity: O(1) per rebalance (single closed-form formula).
"""

import numpy as np
from typing import Dict, Any


def grossman_zhou_allocation(
    current_wealth: float,
    running_max_wealth: float,
    max_drawdown_tolerance: float = 0.15,
    risk_free_rate_annual: float = 0.05,
    expected_excess_return: float = 0.10,
    asset_volatility: float = 0.20,
    risk_aversion: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute the Grossman-Zhou optimal risky allocation under a drawdown
    constraint Wt >= alpha * Mt.

    The closed-form optimal policy invests in proportion to the surplus
    (current wealth minus stochastic floor).

    Args:
        current_wealth: Current portfolio value.
        running_max_wealth: Peak portfolio value (high-water mark).
        max_drawdown_tolerance: Maximum acceptable drawdown fraction (e.g., 0.15 = 15%).
        risk_free_rate_annual: Annualized risk-free rate.
        expected_excess_return: Expected return above risk-free rate.
        asset_volatility: Annualized volatility of risky asset.
        risk_aversion: CRRA risk aversion parameter.

    Returns:
        Dict with optimal risky weight, floor level, surplus, and status.
    """
    alpha = 1.0 - max_drawdown_tolerance  # Floor as fraction of peak
    floor = alpha * running_max_wealth
    surplus = max(current_wealth - floor, 0.0)

    # Merton optimal fraction for unconstrained CRRA investor
    merton_frac = expected_excess_return / (risk_aversion * asset_volatility**2)

    # Grossman-Zhou: invest Merton fraction of surplus, not of total wealth
    if current_wealth > 1e-6:
        risky_weight = merton_frac * (surplus / current_wealth)
        risky_weight = float(np.clip(risky_weight, 0.0, 1.0))
    else:
        risky_weight = 0.0

    drawdown_pct = (
        (running_max_wealth - current_wealth) / running_max_wealth * 100.0
        if running_max_wealth > 0
        else 0.0
    )

    return {
        "risky_weight": round(risky_weight, 4),
        "cash_weight": round(1.0 - risky_weight, 4),
        "floor": round(floor, 2),
        "surplus": round(surplus, 2),
        "current_drawdown_pct": round(drawdown_pct, 2),
        "max_drawdown_tolerance_pct": round(max_drawdown_tolerance * 100, 1),
        "at_floor": bool(surplus < 1e-2),
    }
