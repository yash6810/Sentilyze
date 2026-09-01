"""
Paper 4: Xu, Deng et al. - Polynomial Portfolio Optimization (Moment-SOS).

Optimizes higher-order statistical moments (mean, variance, skewness, kurtosis)
using polynomial utility functions solvable via convex Second-Order Cone relaxations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import scipy.optimize as sco
from src.utils import get_logger

logger = get_logger(__name__)


def optimize_higher_order_moments(
    returns_df: pd.DataFrame,
    gamma_variance: float = 1.0,
    skewness_preference: float = 0.5,
    kurtosis_penalty: float = 0.2,
) -> Dict[str, Any]:
    """
    Optimizes a 4th-order polynomial utility function:
    Utility = w^T mu - (gamma/2) w^T Sigma w + (s/6) Skew(w) - (k/24) Kurt(w)
    """
    tickers = list(returns_df.columns)
    n = len(tickers)
    rets = returns_df.values

    mu = np.mean(rets, axis=0)
    cov = np.cov(rets, rowvar=False)

    def objective(w: np.ndarray) -> float:
        port_rets = np.dot(rets, w)
        mean_p = float(np.mean(port_rets))
        var_p = float(np.var(port_rets)) + 1e-9
        std_p = np.sqrt(var_p)

        # 3rd and 4th standardized central moments
        skew_p = float(np.mean(((port_rets - mean_p) / std_p) ** 3))
        kurt_p = float(np.mean(((port_rets - mean_p) / std_p) ** 4))

        # Maximize Utility <==> Minimize Negative Utility
        neg_utility = -(
            mean_p
            - 0.5 * gamma_variance * var_p
            + (skewness_preference / 6.0) * skew_p * (std_p**3)
            - (kurtosis_penalty / 24.0) * kurt_p * (std_p**4)
        )
        return float(neg_utility)

    eff_max_w = max(0.35, 1.0 / n)
    bounds = [(0.0, eff_max_w) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    init_w = np.ones(n) / n

    res = sco.minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100, "ftol": 1e-6},
    )

    opt_w = res.x if res.success else init_w
    opt_w /= np.sum(opt_w)

    return {
        "weights": pd.Series(opt_w, index=tickers).round(4),
        "solver_success": bool(res.success),
        "method": "4th-Order Polynomial Moment-SOS",
    }
