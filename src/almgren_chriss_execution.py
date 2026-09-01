"""
Paper 3: Almgren & Chriss (2000) - Optimal Execution of Portfolio Transactions.

Computes the exact closed-form optimal liquidation and accumulation trajectory
that minimizes execution shortfall under temporary (eta) and permanent (gamma) price impact.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_almgren_chriss_trajectory(
    total_shares: float,
    total_time_intervals: int = 10,
    daily_volatility: float = 0.02,
    risk_aversion: float = 1e-5,
    temporary_impact_eta: float = 2.5e-6,
    permanent_impact_gamma: float = 2.5e-7,
    initial_price: float = 100.0,
) -> Dict[str, Any]:
    """
    Computes Almgren-Chriss optimal trading trajectory.

    x_j = 2 * sinh(0.5 * kappa * tau) * cosh(kappa * (T - t_j)) / sinh(kappa * T) * X
    """
    N = total_time_intervals
    tau = 1.0  # Normalized unit interval
    T = N * tau
    sigma = daily_volatility
    lambda_param = risk_aversion
    eta = temporary_impact_eta
    gamma = permanent_impact_gamma
    X = total_shares

    # Effective market impact parameter kappa
    # kappa_tilde^2 = (lambda * sigma^2) / eta
    kappa_sq = (lambda_param * (sigma**2)) / (eta + 1e-12)
    kappa = np.sqrt(max(kappa_sq, 1e-10))

    times = np.arange(N + 1)
    # Holdings trajectory x_j remaining at time step j
    holdings = np.zeros(N + 1)
    trades = np.zeros(N)

    for j in range(N + 1):
        t_j = j * tau
        # Analytical hyperbolic solution
        holdings[j] = X * (np.sinh(kappa * (T - t_j)) / (np.sinh(kappa * T) + 1e-12))

    # Trades executed in each step n_j = x_{j-1} - x_j
    for j in range(N):
        trades[j] = holdings[j] - holdings[j + 1]

    # Calculate expected execution cost (Shortfall)
    half_gamma_X2 = 0.5 * gamma * (X**2)
    sum_tau_nj2 = np.sum(trades**2) / tau
    expected_shortfall = half_gamma_X2 + eta * sum_tau_nj2
    shortfall_variance = (sigma**2) * np.sum(holdings[1:] ** 2) * tau

    return {
        "total_shares": float(X),
        "intervals": int(N),
        "holdings_trajectory": [round(float(h), 2) for h in holdings],
        "trade_sizes": [round(float(tr), 2) for tr in trades],
        "expected_shortfall_dollars": round(float(expected_shortfall), 2),
        "shortfall_variance": round(float(shortfall_variance), 2),
        "almgren_kappa": round(float(kappa), 6),
    }
