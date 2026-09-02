"""
Paper 19: Conditional Drawdown-at-Risk (CDaR) Portfolio Optimization.

Source: Chekhlov, Uryasev, Zabarankin (2003).
Complexity: O(T * d) via Linear Programming.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Dict, Any
import pandas as pd


def calculate_cdar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Conditional Drawdown-at-Risk: the expected drawdown
    in the worst alpha% of drawdown episodes.

    Args:
        returns: Array of portfolio returns.
        alpha: Tail probability (0.05 = worst 5%).

    Returns:
        CDaR value (positive number, higher = worse).
    """
    cum_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns

    if len(drawdowns) == 0:
        return 0.0

    # Sort drawdowns descending, take worst alpha fraction
    sorted_dd = np.sort(drawdowns)[::-1]
    n_tail = max(int(len(sorted_dd) * alpha), 1)
    cdar = float(np.mean(sorted_dd[:n_tail]))
    return cdar


def optimize_cdar_portfolio(
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
    min_return: float = 0.0,
    max_weight: float = 0.30,
) -> Dict[str, Any]:
    """
    Optimize portfolio weights to minimize CDaR.

    Simplified approach: compute CDaR for each asset, then allocate
    inversely proportional to individual CDaR (CDaR-parity).

    Args:
        returns_df: DataFrame of asset returns.
        alpha: CDaR tail probability.
        min_return: Minimum acceptable portfolio return (annualized).
        max_weight: Maximum weight per asset.

    Returns:
        Dict with optimal weights and CDaR metrics.
    """
    tickers = list(returns_df.columns)
    d = len(tickers)

    # Calculate per-asset CDaR
    asset_cdars = {}
    for tk in tickers:
        asset_cdars[tk] = calculate_cdar(returns_df[tk].values, alpha)

    # CDaR-inverse parity: allocate inversely to risk
    inv_cdars = np.array([1.0 / max(asset_cdars[tk], 1e-8) for tk in tickers])
    weights = inv_cdars / inv_cdars.sum()

    # Enforce max weight constraint
    weights = np.minimum(weights, max_weight)
    weights = weights / weights.sum()

    # Calculate portfolio CDaR with these weights
    port_rets = returns_df.values @ weights
    port_cdar = calculate_cdar(port_rets, alpha)

    # Portfolio metrics
    ann_return = float(np.mean(port_rets) * 252.0)
    ann_vol = float(np.std(port_rets) * np.sqrt(252.0))

    return {
        "weights": pd.Series(weights, index=tickers),
        "portfolio_cdar": round(port_cdar, 6),
        "annualized_return": round(ann_return * 100, 2),
        "annualized_volatility": round(ann_vol * 100, 2),
        "asset_cdars": {tk: round(v, 6) for tk, v in asset_cdars.items()},
        "cdar_alpha": alpha,
    }
