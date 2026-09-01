"""
Paper 8: HedgeAgents - Balance-Aware Beta & Delta Hedging Engine.

Dynamically computes beta-neutral and tail-risk hedging allocations
between market leaders (Long Alpha) and index hedges (Short Beta).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def compute_balanced_hedge_allocation(
    long_portfolio_returns: pd.Series,
    market_benchmark_returns: pd.Series,
    target_beta: float = 0.0,  # Market-Neutral target beta
    max_hedge_budget_pct: float = 0.20,  # Max 20% allocation to hedge leg
) -> Dict[str, float]:
    """
    Computes the optimal hedge ratio (beta) and hedge asset allocation.
    """
    df_comb = pd.concat(
        [long_portfolio_returns, market_benchmark_returns], axis=1
    ).dropna()
    df_comb.columns = ["port", "bench"]

    cov_matrix = np.cov(df_comb["port"], df_comb["bench"])
    port_var = cov_matrix[0, 0]
    bench_var = cov_matrix[1, 1] + 1e-9
    cov_pb = cov_matrix[0, 1]

    # Current Portfolio Beta to Market
    current_beta = float(cov_pb / bench_var)

    # Required hedge weight to achieve target_beta: w_hedge = (target_beta - beta_p) / beta_hedge
    # Benchmark beta is 1.0
    required_hedge_ratio = float(target_beta - current_beta)
    hedge_weight = float(np.clip(abs(required_hedge_ratio), 0.0, max_hedge_budget_pct))

    return {
        "current_portfolio_beta": round(current_beta, 3),
        "target_beta": round(target_beta, 2),
        "optimal_hedge_weight_pct": round(hedge_weight * 100.0, 2),
        "hedge_instrument": "SPY_INVERSE / SHORT_INDEX_FUTURES",
        "is_hedging_active": bool(hedge_weight > 0.02),
    }
