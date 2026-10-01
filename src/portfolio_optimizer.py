"""
Institutional Portfolio Optimization & Covariance Denoising Engine
(Sprint 1, Module 1.9 / Idea 31)

Features:
1. Ledoit-Wolf Analytical Covariance Shrinkage (Ledoit & Wolf 2004)
2. Random Matrix Theory (RMT) Marchenko-Pastur Eigenvalue Denoising (López de Prado 2018)
3. Minimum Variance Portfolio & Risk-Parity Allocation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.data_ingestion import get_price_history

logger = logging.getLogger("Sentilyze.PortfolioOptimizer")
logging.basicConfig(level=logging.INFO)


def compute_ledoit_wolf_covariance(
    returns_df: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """
    Computes optimal analytical shrinkage covariance matrix using Ledoit-Wolf (2004).
    Returns (shrunk_cov_matrix, shrinkage_intensity).
    """
    if returns_df.empty or returns_df.shape[0] < 5:
        n = returns_df.shape[1] if not returns_df.empty else 1
        return np.eye(n), 0.0

    lw = LedoitWolf()
    lw.fit(returns_df.values)
    cov_lw = lw.covariance_
    shrinkage = float(lw.shrinkage_)
    return cov_lw, round(shrinkage, 4)


def marchenko_pastur_bounds(T: int, N: int, sigma2: float = 1.0) -> Tuple[float, float]:
    """
    Computes theoretical Marchenko-Pastur eigenvalue bounds:
    lambda_max = sigma2 * (1 + sqrt(N / T))^2
    lambda_min = sigma2 * (1 - sqrt(N / T))^2
    """
    if T <= 0 or N <= 0:
        return 0.0, 2.0
    gamma = N / float(T)
    lambda_max = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    lambda_min = sigma2 * (1.0 - np.sqrt(gamma)) ** 2
    return float(lambda_min), float(lambda_max)


def denoise_covariance_rmt(
    cov_matrix: np.ndarray, T: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies Random Matrix Theory (RMT) eigenvalue clipping / shrinkage (López de Prado 2018).
    Separates market signal eigenvalues from random noise eigenvalues.
    """
    N = cov_matrix.shape[0]
    if N <= 1 or T <= N:
        return cov_matrix, {"filtered_modes": 0, "signal_modes": N}

    # Convert covariance to correlation matrix
    diag_std = np.sqrt(np.diag(cov_matrix))
    diag_std_safe = np.where(diag_std > 0, diag_std, 1e-6)
    corr_matrix = cov_matrix / np.outer(diag_std_safe, diag_std_safe)

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Theoretical bounds
    l_min, l_max = marchenko_pastur_bounds(T, N, sigma2=1.0)

    # Identify noise eigenvalues (<= lambda_max)
    noise_mask = eigenvalues <= l_max
    noise_count = int(np.sum(noise_mask))
    signal_count = N - noise_count

    # Equalize noise eigenvalues to their mean
    denoised_evals = eigenvalues.copy()
    if noise_count > 0:
        noise_mean = float(np.mean(eigenvalues[noise_mask]))
        denoised_evals[noise_mask] = noise_mean

    # Reconstruct correlation matrix: C_tilde = V * Lambda_tilde * V^T
    corr_denoised = np.dot(
        eigenvectors, np.dot(np.diag(denoised_evals), eigenvectors.T)
    )

    # Rescale diagonal to 1.0
    diag_corr = np.sqrt(np.diag(corr_denoised))
    diag_corr_safe = np.where(diag_corr > 0, diag_corr, 1e-6)
    corr_denoised = corr_denoised / np.outer(diag_corr_safe, diag_corr_safe)

    # Reconstruct covariance matrix: Sigma_tilde = D^(1/2) * C_tilde * D^(1/2)
    cov_denoised = corr_denoised * np.outer(diag_std_safe, diag_std_safe)

    # Guarantee positive semi-definiteness
    cov_denoised = (cov_denoised + cov_denoised.T) / 2.0
    min_eig = np.min(np.linalg.eigvalsh(cov_denoised))
    if min_eig < 0:
        cov_denoised += np.eye(N) * (-min_eig + 1e-6)

    meta = {
        "lambda_max_mp": round(l_max, 4),
        "lambda_min_mp": round(l_min, 4),
        "total_eigenvalues": N,
        "signal_modes": signal_count,
        "noise_modes_shrunk": noise_count,
        "top_eigenvalue": round(float(eigenvalues[0]), 4),
    }
    return cov_denoised, meta


def compute_minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Computes analytical long-only / non-negative minimum variance portfolio weights:
    w = (Sigma^-1 * 1) / (1^T * Sigma^-1 * 1)
    """
    N = cov_matrix.shape[0]
    if N == 1:
        return np.array([1.0])

    try:
        inv_cov = np.linalg.pinv(cov_matrix)
        ones = np.ones(N)
        raw_weights = np.dot(inv_cov, ones)
        sum_w = float(np.sum(raw_weights))
        if sum_w == 0 or np.isnan(sum_w):
            return np.ones(N) / N
        weights = raw_weights / sum_w

        # Long-only projection (clip negative weights and re-normalize)
        if (weights < 0).any():
            weights = np.clip(weights, 0.0, None)
            total = float(np.sum(weights))
            if total > 0:
                weights = weights / total
            else:
                weights = np.ones(N) / N
        return weights
    except Exception as e:
        logger.debug(f"MVP calculation fallback: {e}")
        return np.ones(N) / N


def optimize_universe_portfolio(
    tickers: List[str], period: str = "6mo"
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Ingests return series for ticker list
    2. Ledoit-Wolf analytical shrinkage
    3. RMT Marchenko-Pastur denoising
    4. Optimal minimum variance portfolio weights
    """
    if not tickers:
        return {"status": "EMPTY_TICKERS", "weights": {}}

    price_series = {}
    for t in tickers:
        try:
            df = get_price_history(t, period=period, use_cache=True)
            if not df.empty and len(df) >= 30:
                price_series[t] = df["Close"]
        except Exception:
            pass

    if len(price_series) < 2:
        eq_w = {t: round(1.0 / len(tickers), 4) for t in tickers}
        return {
            "status": "EQUAL_WEIGHT_FALLBACK",
            "weights": eq_w,
            "message": "Fewer than 2 assets with valid price history.",
        }

    prices_df = pd.DataFrame(price_series).dropna()
    rets_df = prices_df.pct_change().dropna()
    T, N = rets_df.shape

    # 1. Ledoit-Wolf
    cov_lw, shrinkage = compute_ledoit_wolf_covariance(rets_df)

    # 2. RMT Denoising
    cov_denoised, rmt_meta = denoise_covariance_rmt(cov_lw, T=T)

    # 3. Minimum Variance Weights
    weights = compute_minimum_variance_portfolio(cov_denoised)

    weight_dict = {col: round(float(w), 4) for col, w in zip(rets_df.columns, weights)}

    # Annualized portfolio volatility
    port_vol_annual = float(
        np.sqrt(np.dot(weights.T, np.dot(cov_denoised, weights))) * np.sqrt(252)
    )

    return {
        "status": "SUCCESS",
        "tickers": list(rets_df.columns),
        "sample_periods_T": T,
        "assets_N": N,
        "ledoit_wolf_shrinkage": shrinkage,
        "rmt_metadata": rmt_meta,
        "optimized_weights": weight_dict,
        "annualized_portfolio_volatility_pct": round(port_vol_annual * 100.0, 2),
    }


# ==============================================================================
# Sprint 3, Module 3.9 / Idea 32: Conditional Drawdown-at-Risk (CDaR) Allocation
# ==============================================================================
def optimize_cdar_portfolio(
    returns_df: pd.DataFrame,
    alpha: float = 0.95,
    target_return: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Minimizes Conditional Drawdown-at-Risk (CDaR) at confidence level alpha
    (Chekhlov, Uryasev, Zabarankin 2005).

    CDaR_alpha(w) = zeta + (1 / ((1 - alpha) * T)) * sum_t max(D_t(w) - zeta, 0)
    where D_t(w) is the cumulative peak-to-trough drawdown series.
    """
    from scipy.optimize import minimize

    if returns_df.empty or returns_df.shape[1] < 2:
        cols = list(returns_df.columns) if not returns_df.empty else ["ASSET"]
        return {
            "status": "FALLBACK_EQUAL_WEIGHT",
            "weights": {c: 1.0 / len(cols) for c in cols},
            "cdar_pct": 0.0,
        }

    R = returns_df.values
    T, N = R.shape
    mean_rets = np.mean(R, axis=0)

    def cdar_objective(w: np.ndarray) -> float:
        port_rets = np.dot(R, w)
        cum_ret = np.cumsum(port_rets)
        peak = np.maximum.accumulate(cum_ret)
        drawdowns = peak - cum_ret  # Non-negative drawdowns

        # Sample quantile (zeta)
        zeta = float(np.percentile(drawdowns, alpha * 100.0))
        excess = np.maximum(drawdowns - zeta, 0.0)
        cdar = zeta + (1.0 / ((1.0 - alpha) * float(T))) * np.sum(excess)
        return float(cdar)

    # Initial equal weights
    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: np.dot(w, mean_rets) - target_return}
        )

    res = minimize(
        cdar_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    weights = res.x if res.success else w0
    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / np.sum(weights)

    final_cdar = cdar_objective(weights)
    weight_dict = {
        col: round(float(w), 4) for col, w in zip(returns_df.columns, weights)
    }

    return {
        "status": "SUCCESS" if res.success else "APPROX_CONVERGED",
        "alpha": alpha,
        "cdar_drawdown_pct": round(float(final_cdar) * 100.0, 2),
        "weights": weight_dict,
        "expected_daily_return_pct": round(
            float(np.dot(weights, mean_rets)) * 100.0, 4
        ),
    }


# ==============================================================================
# Sprint 3, Module 3.10 / Idea 33: Distributionally Robust Optimization (DRO)
# ==============================================================================
def optimize_dro_portfolio(
    returns_df: pd.DataFrame,
    epsilon: float = 0.05,
    risk_aversion: float = 1.0,
) -> Dict[str, Any]:
    """
    Distributionally Robust Portfolio Optimization over a Wasserstein metric ball (Esfahani & Kuhn 2018).
    Minimizes worst-case mean-variance loss with an L2 norm penalty on weight uncertainty:
    min_{w in Delta} -w^T mu + (gamma / 2) * w^T Sigma w + epsilon * ||w||_2
    """
    from scipy.optimize import minimize

    if returns_df.empty or returns_df.shape[1] < 2:
        cols = list(returns_df.columns) if not returns_df.empty else ["ASSET"]
        return {
            "status": "FALLBACK_EQUAL_WEIGHT",
            "weights": {c: 1.0 / len(cols) for c in cols},
            "wasserstein_radius": epsilon,
        }

    R = returns_df.values
    T, N = R.shape
    mu = np.mean(R, axis=0)
    cov_lw, _ = compute_ledoit_wolf_covariance(returns_df)

    gamma = float(risk_aversion)
    eps = float(epsilon)

    def dro_loss(w: np.ndarray) -> float:
        exp_ret = np.dot(w, mu)
        port_var = np.dot(w.T, np.dot(cov_lw, w))
        robust_penalty = eps * np.linalg.norm(w, ord=2)
        return float(-exp_ret + 0.5 * gamma * port_var + robust_penalty)

    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        dro_loss,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    weights = res.x if res.success else w0
    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / np.sum(weights)

    weight_dict = {
        col: round(float(w), 4) for col, w in zip(returns_df.columns, weights)
    }

    return {
        "status": "SUCCESS" if res.success else "APPROX_CONVERGED",
        "wasserstein_radius_epsilon": eps,
        "risk_aversion_gamma": gamma,
        "weights": weight_dict,
        "robust_objective_val": round(float(res.fun), 6),
    }


# ==============================================================================
# Sprint 3, Module 3.11 / Idea 35: Conformal Black-Litterman (CBL)
# ==============================================================================
def compute_conformal_black_litterman(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    views: Dict[int, float],
    conformal_intervals: Dict[int, Tuple[float, float]],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> Dict[str, Any]:
    """
    Conformal Black-Litterman Engine (Idea 35).
    Dynamically sizes view uncertainty matrix Omega from Conformal Prediction Interval Widths:
    Omega_k,k = ((Upper_k - Lower_k) / (2 * 1.96))^2

    Narrow conformal bounds -> Small Omega -> High view conviction.
    Wide conformal bounds -> Large Omega -> Gracefully shrinks toward equilibrium market prior.
    """
    N = cov_matrix.shape[0]
    w_mkt = np.asarray(market_weights, dtype=float).reshape(N, 1)

    # Implied equilibrium excess returns Pi = lambda * Sigma * w_mkt
    pi = risk_aversion * np.dot(cov_matrix, w_mkt)

    K = len(views)
    if K == 0:
        # No active views -> return market portfolio
        return {
            "status": "NO_VIEWS_MARKET_PRIOR",
            "bl_weights": w_mkt.flatten().tolist(),
            "implied_prior_returns": pi.flatten().tolist(),
            "combined_returns": pi.flatten().tolist(),
        }

    # Construct Pick Matrix P (K x N), View Vector Q (K x 1), and Conformal Omega (K x K)
    P = np.zeros((K, N))
    Q = np.zeros((K, 1))
    omega_diag = np.zeros(K)

    for k, (asset_idx, view_val) in enumerate(views.items()):
        P[k, asset_idx] = 1.0
        Q[k, 0] = view_val

        # Conformal interval width sizing
        if asset_idx in conformal_intervals:
            lower, upper = conformal_intervals[asset_idx]
            width = max(upper - lower, 1e-4)
            # Standard error sigma = width / (2 * 1.96), variance = sigma^2
            omega_diag[k] = (width / 3.92) ** 2
        else:
            # Fallback to He-Litterman proportional variance
            omega_diag[k] = float(tau * np.dot(P[k], np.dot(cov_matrix, P[k].T)))

    Omega = np.diag(omega_diag)

    # Black-Litterman Master Formula:
    # mu_BL = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
    tau_sigma_inv = np.linalg.pinv(tau * cov_matrix)
    omega_inv = np.diag(1.0 / np.maximum(omega_diag, 1e-8))

    m_inv = tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
    M = np.linalg.pinv(m_inv)

    rhs = np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q))
    mu_bl = np.dot(M, rhs)

    # Optimal unconstrained weights: w_BL = (1 / risk_aversion) * Sigma^-1 * mu_bl
    sigma_inv = np.linalg.pinv(cov_matrix)
    raw_weights = (1.0 / risk_aversion) * np.dot(sigma_inv, mu_bl).flatten()

    # Long-only normalization
    weights = np.clip(raw_weights, 0.0, None)
    total_w = float(np.sum(weights))
    if total_w > 0:
        weights = weights / total_w
    else:
        weights = np.ones(N) / N

    return {
        "status": "SUCCESS",
        "n_views": K,
        "conformal_omega_variances": [round(float(v), 6) for v in omega_diag],
        "implied_prior_returns": [round(float(v), 4) for v in pi.flatten()],
        "conformal_bl_returns": [round(float(v), 4) for v in mu_bl.flatten()],
        "bl_weights": [round(float(w), 4) for w in weights],
    }


# ==============================================================================
# Sprint 3, Module 3.12 / Idea 36: Maximum Diversification Ratio (MDR)
# ==============================================================================
def optimize_maximum_diversification_ratio(cov_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Computes Maximum Diversification Ratio (MDR) Portfolio (Choueifaty & Coignard 2008).
    Maximizes Diversification Ratio:
    DR(w) = (w^T * sigma) / sqrt(w^T * Sigma * w)
    where sigma_i = sqrt(Sigma_i,i) is the individual asset volatility.
    """
    from scipy.optimize import minimize

    N = cov_matrix.shape[0]
    if N == 1:
        return {
            "status": "SUCCESS",
            "weights": [1.0],
            "diversification_ratio": 1.0,
        }

    asset_vols = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8))

    def neg_diversification_ratio(w: np.ndarray) -> float:
        weighted_vol = float(np.dot(w, asset_vols))
        port_vol = float(np.sqrt(max(np.dot(w.T, np.dot(cov_matrix, w)), 1e-8)))
        return -(weighted_vol / port_vol)

    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        neg_diversification_ratio,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    weights = res.x if res.success else w0
    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / np.sum(weights)

    max_dr = (
        -float(res.fun) if res.success else float(-neg_diversification_ratio(weights))
    )

    return {
        "status": "SUCCESS" if res.success else "APPROX_CONVERGED",
        "weights": [round(float(w), 4) for w in weights],
        "diversification_ratio": round(max_dr, 4),
        "portfolio_volatility": round(
            float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))), 4
        ),
    }


# ==============================================================================
# Sprint 4, Module 4.6 / Idea 38: Slippage SOCP Rebalancer
# ==============================================================================
def solve_slippage_socp_rebalancing(
    current_weights: Union[np.ndarray, List[float]],
    expected_returns: Union[np.ndarray, List[float]],
    cov_matrix: np.ndarray,
    linear_fee_bps: float = 5.0,
    quadratic_slippage_coeff: float = 0.05,
    risk_aversion_gamma: float = 1.0,
    max_turnover: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Second-Order Cone Programming / Convex Slippage Solver (Boyd et al. 2017).
    Prevents fee churn and transaction drag by solving:
    max_w  mu^T w - (gamma / 2) * w^T Sigma w - sum_i [ c_i * |w_i - w0_i| + kappa_i * (w_i - w0_i)^2 ]
    subject to: 1^T w = 1, w >= 0, and optional turnover <= max_turnover.
    """
    from scipy.optimize import minimize

    w0 = np.asarray(current_weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    N = len(w0)

    # Cost parameters
    c = float(linear_fee_bps) / 10000.0  # Linear fee per dollar traded
    kappa = float(quadratic_slippage_coeff)
    gamma = float(risk_aversion_gamma)

    def neg_net_utility(w: np.ndarray) -> float:
        dw = w - w0
        alpha_term = float(np.dot(mu, w))
        risk_term = float(0.5 * gamma * np.dot(w.T, np.dot(cov_matrix, w)))
        linear_cost = float(c * np.sum(np.abs(dw)))
        slippage_cost = float(kappa * np.sum(dw**2))
        net_obj = alpha_term - risk_term - linear_cost - slippage_cost
        return -net_obj

    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if max_turnover is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: float(max_turnover) - 0.5 * np.sum(np.abs(w - w0)),
            }
        )

    res = minimize(
        neg_net_utility,
        w0.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-7},
    )

    opt_w = res.x if res.success else w0
    # Clean tiny noise churn (< 0.5 bps)
    delta_w = opt_w - w0
    pruned_delta = np.where(np.abs(delta_w) < 0.0005, 0.0, delta_w)
    final_w = w0 + pruned_delta
    final_w = np.clip(final_w, 0.0, None)
    total = np.sum(final_w)
    final_w = final_w / total if total > 0 else np.ones(N) / N

    final_delta = final_w - w0
    turnover = float(0.5 * np.sum(np.abs(final_delta)))
    linear_fee_cost = float(c * np.sum(np.abs(final_delta))) * 10000.0  # in bps
    slippage_cost = float(kappa * np.sum(final_delta**2)) * 10000.0  # in bps
    churn_suppressed = bool((np.abs(delta_w) < 0.0005).any())

    return {
        "status": "SUCCESS" if res.success else "APPROX_CONVERGED",
        "optimal_weights": [round(float(w), 4) for w in final_w],
        "delta_weights": [round(float(dw), 4) for dw in final_delta],
        "portfolio_turnover": round(turnover, 4),
        "linear_fee_cost_bps": round(linear_fee_cost, 2),
        "slippage_cost_bps": round(slippage_cost, 2),
        "total_transaction_drag_bps": round(linear_fee_cost + slippage_cost, 2),
        "churn_suppressed": churn_suppressed,
    }
