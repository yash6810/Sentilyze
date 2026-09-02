"""
Paper 23: Risk-Constrained Kelly Gambling.

Source: Busseti, Ryu, Boyd — Stanford, Journal of Investing, 2016.
Complexity: O(d^3) convex solver.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Optional
import pandas as pd


def risk_constrained_kelly_allocation(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_drawdown_prob: float = 0.05,
    max_drawdown_level: float = 0.15,
    risk_free_rate: float = 0.0,
    max_leverage: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute optimal Kelly allocation with drawdown probability constraint.

    Maximizes E[log(1 + w'r)] subject to P(drawdown > d) <= epsilon,
    approximated as a convex program.

    Args:
        expected_returns: Expected excess returns for each asset (annualized).
        cov_matrix: Covariance matrix (annualized).
        max_drawdown_prob: Maximum probability of exceeding drawdown (epsilon).
        max_drawdown_level: Maximum drawdown level (d).
        risk_free_rate: Risk-free rate.
        max_leverage: Maximum total weight (1.0 = fully invested).

    Returns:
        Dict with optimal weights and risk metrics.
    """
    d = len(expected_returns)
    mu = expected_returns - risk_free_rate
    Sigma = cov_matrix

    # Risk constraint: w'Sigma w <= variance_budget
    # From Busseti et al.: variance budget derived from drawdown constraint
    # P(DD > d) <= exp(-2 * d^2 / (w'Sigma w * T))
    # => w'Sigma w <= -2 * d^2 / (T * log(epsilon))
    # For daily rebalancing, T ~ 252
    T = 252.0
    log_eps = np.log(max(max_drawdown_prob, 1e-10))
    variance_budget = -2.0 * max_drawdown_level**2 / (T * log_eps)
    variance_budget = max(variance_budget, 1e-8)

    def neg_log_growth(w):
        port_ret = float(np.dot(w, mu))
        port_var = float(w @ Sigma @ w)
        # Approximate log growth: mu'w - 0.5 * w'Sigma w
        return -(port_ret - 0.5 * port_var)

    # Constraints
    constraints = [
        {
            "type": "ineq",
            "fun": lambda w: max_leverage - np.sum(w),
        },  # sum(w) <= max_leverage
        {
            "type": "ineq",
            "fun": lambda w: variance_budget - float(w @ Sigma @ w),
        },  # risk constraint
    ]

    bounds = [(0.0, max_leverage)] * d
    w0 = np.ones(d) / d * max_leverage * 0.5  # Conservative start

    result = minimize(
        neg_log_growth,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-10},
    )

    weights = result.x if result.success else w0
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total > max_leverage:
        weights = weights / total * max_leverage

    port_ret = float(np.dot(weights, mu))
    port_var = float(weights @ Sigma @ weights)
    log_growth = port_ret - 0.5 * port_var
    sharpe = port_ret / max(np.sqrt(port_var), 1e-9)

    return {
        "weights": weights,
        "log_growth_rate": round(log_growth, 6),
        "expected_return": round(port_ret, 4),
        "portfolio_variance": round(port_var, 6),
        "portfolio_sharpe": round(float(sharpe), 2),
        "variance_budget": round(variance_budget, 6),
        "variance_used_pct": (
            round(port_var / variance_budget * 100, 2) if variance_budget > 0 else 0.0
        ),
        "solver_converged": bool(result.success),
    }
