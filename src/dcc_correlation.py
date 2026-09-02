"""
Paper 24: Dynamic Conditional Correlation (DCC-GARCH).

Source: Engle (2002) — "Dynamic Conditional Correlation", JBES.
Complexity: O(d^2 * T).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class DCCCorrelation:
    """
    Simplified DCC model using EWMA-GARCH(1,1) for individual
    volatilities and DCC evolution for correlations.

    Two-step estimation:
    1. Univariate GARCH(1,1) for each asset volatility.
    2. DCC correlation evolution from standardized residuals.
    """

    def __init__(
        self,
        garch_omega: float = 0.00001,
        garch_alpha: float = 0.06,
        garch_beta: float = 0.93,
        dcc_a: float = 0.05,
        dcc_b: float = 0.93,
    ):
        self.omega = garch_omega
        self.alpha_g = garch_alpha
        self.beta_g = garch_beta
        self.dcc_a = dcc_a
        self.dcc_b = dcc_b

    def fit(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the DCC model to a returns DataFrame.

        Returns time-varying correlation matrices and regime alerts.
        """
        tickers = list(returns_df.columns)
        d = len(tickers)
        T = len(returns_df)
        R = returns_df.values  # T x d

        # Step 1: Univariate GARCH(1,1) for each asset
        h = np.zeros((T, d))  # Conditional variances
        h[0] = np.var(R[:20], axis=0) if T >= 20 else np.var(R, axis=0) + 1e-8

        for t in range(1, T):
            h[t] = self.omega + self.alpha_g * R[t - 1] ** 2 + self.beta_g * h[t - 1]
            h[t] = np.maximum(h[t], 1e-10)

        # Standardized residuals
        eps = R / np.sqrt(h)

        # Step 2: DCC correlation evolution
        Q_bar = np.corrcoef(eps.T)  # Unconditional correlation of std residuals
        if np.any(np.isnan(Q_bar)):
            Q_bar = np.eye(d)

        Q_t = Q_bar.copy()
        corr_series = np.zeros((T, d, d))
        avg_corr_series = np.zeros(T)

        for t in range(T):
            if t > 0:
                outer = np.outer(eps[t - 1], eps[t - 1])
                Q_t = (
                    (1 - self.dcc_a - self.dcc_b) * Q_bar
                    + self.dcc_a * outer
                    + self.dcc_b * Q_t
                )

            # Normalize Q_t to get correlation matrix R_t
            diag_inv = 1.0 / np.sqrt(np.maximum(np.diag(Q_t), 1e-10))
            R_t = Q_t * np.outer(diag_inv, diag_inv)
            np.fill_diagonal(R_t, 1.0)
            R_t = np.clip(R_t, -1.0, 1.0)

            corr_series[t] = R_t

            if d > 1:
                mask = ~np.eye(d, dtype=bool)
                avg_corr_series[t] = np.mean(R_t[mask])

        # Final state
        final_corr = corr_series[-1]
        final_avg = float(avg_corr_series[-1])

        # Annualized volatilities from GARCH
        ann_vols = {
            tk: round(float(np.sqrt(h[-1, i] * 252.0)) * 100, 2)
            for i, tk in enumerate(tickers)
        }

        return {
            "final_correlation_matrix": pd.DataFrame(
                final_corr, index=tickers, columns=tickers
            ),
            "final_avg_pairwise_correlation": round(final_avg, 4),
            "correlation_breakdown_alert": bool(final_avg > 0.75),
            "annualized_garch_volatilities": ann_vols,
            "avg_correlation_timeseries": avg_corr_series,
            "n_observations": T,
        }
