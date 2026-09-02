"""
Paper 17: RiskMetrics EWMA Volatility & Correlation Monitor.

Source: J.P. Morgan RiskMetrics (1996).
Complexity: O(1) per observation per asset pair (constant time).
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


class EWMACorrelationMonitor:
    """
    Real-time EWMA-based correlation and volatility monitor.

    Tracks time-varying correlations between assets using the recursive
    EWMA formula. Raises alert when average pairwise correlation exceeds
    a breakdown threshold (default 0.75).

    Parameters:
        decay_lambda: EWMA decay factor (0.94 = RiskMetrics standard).
        correlation_alert_threshold: Alert when avg correlation exceeds this.
    """

    def __init__(
        self,
        decay_lambda: float = 0.94,
        correlation_alert_threshold: float = 0.75,
    ):
        self.lam = decay_lambda
        self.alert_threshold = correlation_alert_threshold
        self.cov_matrix: Optional[np.ndarray] = None
        self.var_vector: Optional[np.ndarray] = None
        self.tickers: list = []
        self.initialized = False

    def initialize(self, returns_df: pd.DataFrame):
        """Initialize EWMA state from a seed window of returns."""
        self.tickers = list(returns_df.columns)
        d = len(self.tickers)
        seed_cov = returns_df.cov().values
        self.cov_matrix = seed_cov.copy()
        self.var_vector = np.diag(seed_cov).copy()
        self.initialized = True

    def update(self, returns_vector: np.ndarray) -> Dict[str, Any]:
        """
        Update EWMA state with one day's returns across all assets.
        Returns current correlation matrix and alert status.
        """
        if not self.initialized:
            raise RuntimeError("Call initialize() with seed data first.")

        r = returns_vector.reshape(-1, 1)
        outer = r @ r.T

        self.cov_matrix = self.lam * self.cov_matrix + (1.0 - self.lam) * outer
        self.var_vector = self.lam * self.var_vector + (1.0 - self.lam) * (
            returns_vector**2
        )

        # Derive correlation matrix
        std_vec = np.sqrt(np.maximum(self.var_vector, 1e-12))
        std_outer = np.outer(std_vec, std_vec)
        corr_matrix = self.cov_matrix / std_outer
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # Average off-diagonal correlation
        d = len(self.tickers)
        if d > 1:
            mask = ~np.eye(d, dtype=bool)
            avg_corr = float(np.mean(corr_matrix[mask]))
        else:
            avg_corr = 0.0

        alert = avg_corr > self.alert_threshold

        return {
            "avg_pairwise_correlation": round(avg_corr, 4),
            "correlation_breakdown_alert": alert,
            "ewma_volatilities": {
                tk: round(float(np.sqrt(v * 252.0)) * 100.0, 2)
                for tk, v in zip(self.tickers, self.var_vector)
            },
        }

    def get_ewma_covariance(self) -> np.ndarray:
        """Return current annualized EWMA covariance matrix."""
        return self.cov_matrix * 252.0
