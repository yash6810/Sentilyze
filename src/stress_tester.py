import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from src.utils import get_logger

logger = get_logger(__name__)


def run_monte_carlo_stress_test(
    initial_capital: float = 100000.0,
    daily_returns_df: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    num_simulations: int = 1000,
    time_horizon_days: int = 30,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Runs an institutional Monte Carlo forward stress test and Value-at-Risk (VaR) simulation.

    Args:
        initial_capital (float): Current portfolio equity.
        daily_returns_df (pd.DataFrame, optional): Historical daily returns DataFrame across assets.
        weights (Dict[str, float], optional): Allocation weights per asset.
        num_simulations (int): Number of simulated future paths (default: 1,000).
        time_horizon_days (int): Future simulation horizon in trading days (default: 30).
        confidence_level (float): Confidence level for VaR (default: 0.95).

    Returns:
        Dict[str, Any]: Comprehensive risk metrics, VaR, CVaR, and simulation path matrix.
    """
    if daily_returns_df is not None and not daily_returns_df.empty and weights:
        # Portfolio daily return series from historical covariance
        active_assets = [col for col in daily_returns_df.columns if col in weights]
        if active_assets:
            w_vec = np.array([weights[a] for a in active_assets])
            w_vec = w_vec / np.sum(w_vec)
            sub_returns = daily_returns_df[active_assets].dropna()
            port_returns = sub_returns.dot(w_vec)
            mu = port_returns.mean()
            sigma = port_returns.std()
        else:
            mu, sigma = 0.0008, 0.012  # Annualized ~20% return, 19% vol
    else:
        # Default market empirical parameters for a balanced tech/equity quant portfolio
        mu, sigma = 0.0008, 0.012

    # Geometric Brownian Motion simulation
    # S_t = S_0 * exp(cumsum((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z))
    dt = 1.0
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    vol_step = sigma * np.sqrt(dt)

    # Random shocks: shape (num_simulations, time_horizon_days)
    np.random.seed(42)
    shocks = np.random.normal(0, 1, size=(num_simulations, time_horizon_days))
    daily_growth = np.exp(drift + vol_step * shocks)

    # Compute price paths starting from initial_capital
    paths = np.zeros((num_simulations, time_horizon_days + 1))
    paths[:, 0] = initial_capital
    paths[:, 1:] = initial_capital * np.cumprod(daily_growth, axis=1)

    # Compute metrics on final day outcomes
    final_equities = paths[:, -1]
    net_pnls = final_equities - initial_capital
    returns_pct = (final_equities - initial_capital) / initial_capital * 100.0

    # Value at Risk (VaR) & Conditional VaR (Expected Shortfall)
    # VaR_95 is the loss at the (1 - confidence_level) percentile
    alpha_percentile = (1.0 - confidence_level) * 100.0
    var_dollar = float(np.percentile(-net_pnls, confidence_level * 100.0))
    var_pct = float(np.percentile(-returns_pct, confidence_level * 100.0))

    # CVaR (Expected Shortfall) = Average loss beyond VaR cutoff
    tail_losses = -net_pnls[-net_pnls >= var_dollar]
    cvar_dollar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_dollar

    # Maximum simulated drawdown across all paths
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max * 100.0
    max_drawdown_5th_pct = float(np.percentile(np.min(drawdowns, axis=1), 5))

    prob_profit = float(np.mean(final_equities > initial_capital) * 100.0)

    # Summary quantiles for visual plotting
    percentiles = {
        "5th_worst": np.percentile(paths, 5, axis=0),
        "25th_pct": np.percentile(paths, 25, axis=0),
        "50th_median": np.percentile(paths, 50, axis=0),
        "75th_pct": np.percentile(paths, 75, axis=0),
        "95th_best": np.percentile(paths, 95, axis=0),
    }

    df_percentiles = pd.DataFrame(
        percentiles,
        index=[f"Day {i}" for i in range(time_horizon_days + 1)],
    )

    logger.info(
        f"Monte Carlo stress test complete ({num_simulations} paths over {time_horizon_days} days). "
        f"95% VaR: ${var_dollar:,.2f} ({var_pct:.2f}%) | Prob Profit: {prob_profit:.1f}%"
    )

    return {
        "initial_capital": initial_capital,
        "time_horizon_days": time_horizon_days,
        "num_simulations": num_simulations,
        "var_95_dollar": round(var_dollar, 2),
        "var_95_pct": round(var_pct, 2),
        "cvar_95_dollar": round(cvar_dollar, 2),
        "median_final_equity": round(float(np.median(final_equities)), 2),
        "expected_return_pct": round(float(np.mean(returns_pct)), 2),
        "prob_profit": round(prob_profit, 1),
        "worst_case_drawdown_pct": round(max_drawdown_5th_pct, 2),
        "percentile_paths_df": df_percentiles,
    }


def run_monte_carlo_var(
    initial_equity: float = 100000.0,
    num_paths: int = 1000,
    days: int = 30,
) -> Dict[str, Any]:
    """Helper wrapper for Monte Carlo VaR simulation."""
    res = run_monte_carlo_stress_test(
        initial_capital=initial_equity,
        num_simulations=num_paths,
        time_horizon_days=days,
    )
    res["prob_profit_pct"] = res.get("prob_profit", 65.0)
    return res

