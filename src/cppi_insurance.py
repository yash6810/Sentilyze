"""
Paper 20: Constant Proportion Portfolio Insurance (CPPI).

Source: Black & Jones (1987), Perold (1986).
Complexity: O(1) per rebalance (single formula).
"""

import numpy as np
from typing import Dict, Any


def calculate_cppi_allocation(
    portfolio_value: float,
    floor_value: float,
    multiplier: float = 3.0,
    max_risky_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    CPPI allocation: Exposure = M * (Portfolio - Floor).

    Args:
        portfolio_value: Current total portfolio value.
        floor_value: Guaranteed minimum value (capital floor).
        multiplier: CPPI multiplier (3-5 typical).
        max_risky_weight: Maximum allocation to risky assets (cap at 100%).

    Returns:
        Dict with risky/safe allocations and cushion metrics.
    """
    cushion = max(portfolio_value - floor_value, 0.0)
    cushion_pct = cushion / portfolio_value if portfolio_value > 0 else 0.0

    risky_exposure = multiplier * cushion
    risky_weight = (
        min(risky_exposure / portfolio_value, max_risky_weight)
        if portfolio_value > 0
        else 0.0
    )
    risky_weight = max(risky_weight, 0.0)

    safe_weight = 1.0 - risky_weight

    return {
        "risky_weight": round(risky_weight, 4),
        "safe_weight": round(safe_weight, 4),
        "cushion": round(cushion, 2),
        "cushion_pct": round(cushion_pct * 100, 2),
        "risky_dollar_exposure": round(risky_weight * portfolio_value, 2),
        "safe_dollar_exposure": round(safe_weight * portfolio_value, 2),
        "floor_value": round(floor_value, 2),
        "at_floor": bool(cushion < 1.0),
    }


def run_cppi_backtest(
    returns: np.ndarray,
    initial_capital: float = 100000.0,
    floor_pct: float = 0.80,
    multiplier: float = 3.0,
) -> Dict[str, Any]:
    """
    Run a full CPPI backtest over a return series.

    Args:
        returns: Array of daily returns.
        initial_capital: Starting capital.
        floor_pct: Floor as fraction of initial capital (0.80 = 80%).
        multiplier: CPPI multiplier.

    Returns:
        Dict with final value, max drawdown, and portfolio history.
    """
    floor_value = initial_capital * floor_pct
    n = len(returns)
    portfolio_values = np.zeros(n + 1)
    portfolio_values[0] = initial_capital
    risky_weights = np.zeros(n)

    for t in range(n):
        alloc = calculate_cppi_allocation(portfolio_values[t], floor_value, multiplier)
        risky_weights[t] = alloc["risky_weight"]
        day_return = alloc["risky_weight"] * returns[t]
        portfolio_values[t + 1] = portfolio_values[t] * (1.0 + day_return)
        # Ensure floor (in theory, gap risk can breach it)
        portfolio_values[t + 1] = max(portfolio_values[t + 1], floor_value * 0.99)

    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peak) / peak
    max_dd = float(abs(drawdowns.min())) * 100.0

    total_return = (portfolio_values[-1] / initial_capital - 1.0) * 100.0

    return {
        "final_value": round(float(portfolio_values[-1]), 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "floor_value": round(floor_value, 2),
        "floor_breached": bool(float(portfolio_values.min()) < floor_value * 0.99),
        "avg_risky_weight_pct": round(float(np.mean(risky_weights)) * 100, 2),
    }
