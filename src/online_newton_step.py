"""
Online Newton Step (ONS) Portfolio Engine (Agarwal, Hazan, Kale).

A polynomial-time Online Convex Optimization (OCO) algorithm:
- Achieves optimal logarithmic regret O(d * log(T))
- Computes per-round portfolio updates in O(d^2) time via Sherman-Morrison rank-1 updates
- Dynamically adapts to curvature of the logarithmic wealth loss function
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


class OnlineNewtonStepOptimizer:
    """
    Polynomial-Time ONS Portfolio Engine (Hazan et al.).
    """

    def __init__(self, num_assets: int, eta: float = 0.5, beta: float = 1.0):
        self.d = num_assets
        self.eta = eta
        self.beta = beta
        # Initialize inverse Hessian approximation A^-1 = (1/eps) * I
        self.eps = 1.0 / (self.d**2)
        self.A_inv = np.eye(self.d) / self.eps
        # Start with uniform simplex weights
        self.w = np.ones(self.d) / self.d

    def step(self, asset_price_relatives: np.ndarray) -> np.ndarray:
        """
        Processes price relatives (r_t = Close_t / Close_{t-1}) and updates weights in O(d^2) time.

        Args:
            asset_price_relatives: Array of 1 + returns for day t

        Returns:
            Updated weight vector w_{t+1}
        """
        r = np.array(asset_price_relatives, dtype=float)
        # Portfolio return
        port_ret = float(np.dot(self.w, r))
        if port_ret < 1e-6:
            return self.w

        # Gradient of loss f_t(w) = -ln(w^T r) is -r / (w^T r)
        grad = -r / port_ret

        # Sherman-Morrison Rank-1 Inverse Hessian Update:
        # (A + g g^T)^-1 = A^-1 - (A^-1 g g^T A^-1) / (1 + g^T A^-1 g)
        A_inv_g = np.dot(self.A_inv, grad)
        denom = 1.0 + float(np.dot(grad, A_inv_g))
        self.A_inv -= np.outer(A_inv_g, A_inv_g) / denom

        # Newton step
        w_tilde = self.w - (1.0 / self.beta) * np.dot(self.A_inv, grad)

        # Simplex Projection (Project w_tilde onto sum(w) = 1, w >= 0)
        self.w = self._project_to_simplex(w_tilde)
        return self.w

    def _project_to_simplex(self, v: np.ndarray) -> np.ndarray:
        """Fast O(d log d) Euclidean projection onto probability simplex."""
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, self.d + 1) > (cssv - 1.0))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        w_proj = np.maximum(v - theta, 0.0)
        norm_sum = np.sum(w_proj)
        if norm_sum > 0:
            return w_proj / norm_sum
        return np.ones(self.d) / self.d

    def backtest_sequence(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs ONS sequence through time and outputs daily allocations and portfolio returns.
        """
        start_time = time.perf_counter()
        rets_matrix = returns_df.values
        n_days, n_assets = rets_matrix.shape
        tickers = list(returns_df.columns)

        daily_port_rets = np.zeros(n_days)
        weights_history = np.zeros((n_days, n_assets))

        for t in range(n_days):
            r_t = 1.0 + rets_matrix[t]  # price relatives
            weights_history[t] = self.w.copy()
            daily_port_rets[t] = float(np.dot(self.w, rets_matrix[t]))
            self.step(r_t)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        res_df = pd.DataFrame(index=returns_df.index)
        res_df["daily_return"] = daily_port_rets
        res_df["total"] = (1.0 + daily_port_rets).cumprod()

        logger.info(
            f"ONS Sequential Run Completed: {n_days} days across {n_assets} assets in {elapsed_ms:.2f}ms"
        )
        return res_df
