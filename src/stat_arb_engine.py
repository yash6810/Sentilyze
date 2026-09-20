"""Statistical Arbitrage & Cointegration Engine for Sentilyze.

Module: src/stat_arb_engine.py

Provides:
- Vectorized Engle-Granger & Total Least Squares (TLS) cointegration tests
- Continuous-time Ornstein-Uhlenbeck (OU) SDE parameter calibration
- Half-life of mean reversion computation (t_1/2)
- Rolling Z-score spread tracking with dynamic Bollinger Bands (+-2.0 sigma, +-3.5 sigma)
- Regime detection via Hurst Exponent (R/S analysis)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def compute_hurst_exponent(price_series: np.ndarray, max_lag: int = 50) -> float:
    """Computes the Hurst Exponent (H) using Rescaled Range (R/S) analysis.

    H > 0.55: Persistent / Trending (Momentum favored)
    H < 0.45: Anti-Persistent / Mean-Reverting (Stat-Arb favored)
    H ~ 0.50: Random Walk / Brownian Motion
    """
    clean_series = price_series[~np.isnan(price_series)]
    if len(clean_series) < max_lag * 2:
        return 0.50

    rets = np.diff(np.log(clean_series))
    if len(rets) < 20:
        return 0.50

    lags = range(10, min(max_lag, len(rets) // 2))
    tau = []
    rs_vals = []

    for lag in lags:
        n_chunks = len(rets) // lag
        if n_chunks < 2:
            continue
        chunks = rets[: n_chunks * lag].reshape((n_chunks, lag))

        means = np.mean(chunks, axis=1, keepdims=True)
        devs = np.cumsum(chunks - means, axis=1)
        ranges = np.ptp(devs, axis=1)
        stds = np.std(chunks, axis=1, ddof=1) + 1e-10

        rs = np.mean(ranges / stds)
        if rs > 0:
            tau.append(lag)
            rs_vals.append(rs)

    if len(tau) < 3:
        return 0.50

    try:
        poly = np.polyfit(np.log(tau), np.log(rs_vals), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.50


def fit_ornstein_uhlenbeck(spread: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
    """Fits continuous-time Ornstein-Uhlenbeck SDE:
        dS_t = theta * (mu - S_t) dt + sigma * dW_t
    via exact discrete AR(1) linear regression:
        S_t = a * S_{t-1} + b + eps

    Returns:
        theta: Mean-reversion speed
        mu: Equilibrium spread level
        sigma: Diffusion volatility
        half_life: Expected trading days to revert 50% of deviation
    """
    clean_spread = spread[~np.isnan(spread)]
    if len(clean_spread) < 20:
        return {"theta": 0.0, "mu": 0.0, "sigma": 0.0, "half_life": 999.0}

    s_curr = clean_spread[1:]
    s_prev = clean_spread[:-1]

    # OLS regression of S_t on S_{t-1}
    slope, intercept, _, _, _ = stats.linregress(s_prev, s_curr)

    # Unit root or explosive check
    if slope >= 1.0 or slope <= 0.0:
        return {
            "theta": 0.0,
            "mu": float(np.mean(clean_spread)),
            "sigma": float(np.std(clean_spread)),
            "half_life": 999.0,
        }

    theta = float(-np.log(slope) / dt)
    mu = float(intercept / (1.0 - slope))

    residuals = s_curr - (slope * s_prev + intercept)
    var_eps = float(np.var(residuals, ddof=2))

    denom = 1.0 - np.exp(-2.0 * theta * dt)
    sigma = float(np.sqrt(var_eps * 2.0 * theta / max(denom, 1e-10)))
    half_life = float(np.log(2.0) / max(theta, 1e-10))

    return {
        "theta": round(theta, 5),
        "mu": round(mu, 5),
        "sigma": round(sigma, 5),
        "half_life": round(min(half_life, 999.0), 2),
    }


def test_pairs_cointegration(p1: np.ndarray, p2: np.ndarray) -> Dict[str, Any]:
    """Performs Engle-Granger two-step cointegration test with Total Least Squares
    (TLS) hedge ratio and Augmented Dickey-Fuller p-value.
    """
    clean_mask = ~(np.isnan(p1) | np.isnan(p2))
    s1 = p1[clean_mask]
    s2 = p2[clean_mask]

    if len(s1) < 30:
        return {
            "beta": 1.0,
            "alpha": 0.0,
            "adf_stat": 0.0,
            "p_value": 1.0,
            "is_cointegrated": False,
            "crit_1pct": -3.5,
            "crit_5pct": -2.9,
            "spread_series": np.zeros(len(s1)),
            "ou_params": {"theta": 0.0, "mu": 0.0, "sigma": 0.0, "half_life": 999.0},
        }

    log_p1 = np.log(np.maximum(s1, 1e-6))
    log_p2 = np.log(np.maximum(s2, 1e-6))

    # 1. Total Least Squares (TLS) for Symmetric Hedge Ratio
    try:
        cov_mat = np.cov(log_p2, log_p1)
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        tls_vector = eigvecs[:, np.argmax(eigvals)]
        beta = float(tls_vector[1] / (tls_vector[0] + 1e-10))
        if abs(beta) > 10.0 or abs(beta) < 0.01:
            # Fallback to standard OLS
            slope, intercept, _, _, _ = stats.linregress(log_p2, log_p1)
            beta = float(slope)
            alpha = float(intercept)
        else:
            alpha = float(np.mean(log_p1) - beta * np.mean(log_p2))
    except Exception:
        slope, intercept, _, _, _ = stats.linregress(log_p2, log_p1)
        beta = float(slope)
        alpha = float(intercept)

    # 2. Stationary Spread Series S_t = ln(P1) - beta*ln(P2) - alpha
    spread = log_p1 - beta * log_p2 - alpha

    # 3. Augmented Dickey-Fuller Test on Spread
    try:
        adf_res = adfuller(spread, maxlag=5, autolag="AIC")
        adf_stat = float(adf_res[0])
        p_value = float(adf_res[1])
        crit_vals = adf_res[4]
    except Exception:
        adf_stat = 0.0
        p_value = 1.0
        crit_vals = {"1%": -3.5, "5%": -2.9}

    # 4. Fit Continuous-Time OU SDE
    ou_params = fit_ornstein_uhlenbeck(spread)

    return {
        "beta": round(beta, 4),
        "alpha": round(alpha, 4),
        "adf_stat": round(adf_stat, 3),
        "p_value": round(p_value, 5),
        "is_cointegrated": bool(p_value < 0.05 and ou_params["half_life"] < 60.0),
        "crit_1pct": round(crit_vals.get("1%", -3.5), 3),
        "crit_5pct": round(crit_vals.get("5%", -2.9), 3),
        "spread_series": spread,
        "ou_params": ou_params,
    }


def compute_dynamic_spread_zscore(
    spread: np.ndarray,
    half_life: float,
    min_window: int = 20,
    max_window: int = 60,
) -> Dict[str, Any]:
    """Computes rolling Z-scores and adaptive Bollinger Bands scaled by OU half-life."""
    valid_hl = half_life if (0 < half_life < 999.0) else 20.0
    window = int(np.clip(round(2.0 * valid_hl), min_window, max_window))
    s_series = pd.Series(spread)

    roll_mean = s_series.rolling(window=window).mean()
    roll_std = s_series.rolling(window=window).std()

    z_score = (s_series - roll_mean) / (roll_std + 1e-10)

    curr_z = float(z_score.iloc[-1]) if not np.isnan(z_score.iloc[-1]) else 0.0
    curr_mean = float(roll_mean.iloc[-1]) if not np.isnan(roll_mean.iloc[-1]) else 0.0
    curr_std = float(roll_std.iloc[-1]) if not np.isnan(roll_std.iloc[-1]) else 0.0

    return {
        "window_used": window,
        "current_zscore": round(curr_z, 2),
        "rolling_mean": round(curr_mean, 5),
        "rolling_std": round(curr_std, 5),
        "bollinger_upper_2sigma": round(curr_mean + 2.0 * curr_std, 5),
        "bollinger_lower_2sigma": round(curr_mean - 2.0 * curr_std, 5),
        "stop_loss_upper_3_5sigma": round(curr_mean + 3.5 * curr_std, 5),
        "stop_loss_lower_3_5sigma": round(curr_mean - 3.5 * curr_std, 5),
        "zscore_series": z_score.fillna(0.0).values,
    }
