"""
Marcos López de Prado's Triple-Barrier Method & Deflated Sharpe Ratio (DSR).

Pillars:
1. Triple-Barrier Labeling:
   - Dynamic upper profit barrier (e.g. +2.0 ATR)
   - Dynamic lower stop-loss barrier (e.g. -1.5 ATR)
   - Vertical time expiration barrier (e.g. 5 days)
2. Deflated Sharpe Ratio (DSR):
   - Corrects for backtest overfitting, selection bias, non-normality (skewness/kurtosis),
     and multiple testing trials.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from src.utils import get_logger

logger = get_logger(__name__)


def apply_triple_barrier_labeling(
    df: pd.DataFrame,
    profit_taking_mult: float = 2.0,
    stop_loss_mult: float = 1.5,
    max_holding_days: int = 5,
) -> pd.DataFrame:
    """
    Applies López de Prado's path-dependent Triple-Barrier Method to generate trade labels.

    Args:
        df: Price history with High, Low, Close
        profit_taking_mult: Multiplier of ATR for upper profit barrier
        stop_loss_mult: Multiplier of ATR for lower stop-loss barrier
        max_holding_days: Maximum holding period (vertical barrier)

    Returns:
        pd.DataFrame with 'target_barrier' (+1 = Profit Target, -1 = Stop Loss, 0 = Time Out)
    """
    df_res = df.copy()
    high = df_res["High"]
    low = df_res["Low"]
    close = df_res["Close"]

    # Compute 14-day True Range / Volatility
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().fillna(tr.mean())

    n = len(df_res)
    labels = np.zeros(n, dtype=int)
    exit_ret = np.zeros(n, dtype=float)

    close_vals = close.values
    high_vals = high.values
    low_vals = low.values
    atr_vals = atr.values

    for i in range(n - max_holding_days):
        entry_price = close_vals[i]
        vol = atr_vals[i]
        upper_barrier = entry_price + profit_taking_mult * vol
        lower_barrier = entry_price - stop_loss_mult * vol

        hit_upper = False
        hit_lower = False

        for step in range(1, max_holding_days + 1):
            curr_high = high_vals[i + step]
            curr_low = low_vals[i + step]

            if curr_high >= upper_barrier:
                hit_upper = True
                exit_ret[i] = (upper_barrier - entry_price) / entry_price
                break
            elif curr_low <= lower_barrier:
                hit_lower = True
                exit_ret[i] = (lower_barrier - entry_price) / entry_price
                break

        if hit_upper:
            labels[i] = 1
        elif hit_lower:
            labels[i] = -1
        else:
            labels[i] = 0
            # Return at time expiration
            exit_ret[i] = (close_vals[i + max_holding_days] - entry_price) / entry_price

    df_res["target_barrier"] = labels
    df_res["barrier_return"] = exit_ret
    return df_res


def calculate_deflated_sharpe_ratio(
    strategy_returns: pd.Series,
    num_trials: int = 50,
    benchmark_sharpe: float = 0.0,
) -> Dict[str, float]:
    """
    Computes Bailey & López de Prado's Deflated Sharpe Ratio (DSR).

    Adjusts for:
    - Multiple testing selection bias (num_trials)
    - Non-normality (skewness and excess kurtosis)
    - Sample track record length (T)

    Returns:
        Dict with annualized_sharpe, dsr_pvalue, and is_statistically_significant.
    """
    rets = strategy_returns.dropna().values
    T = len(rets)
    if T < 15:
        return {
            "annualized_sharpe": 0.0,
            "dsr_prob": 0.0,
            "is_statistically_significant": False,
        }

    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) + 1e-9
    daily_sr = mean_ret / std_ret
    ann_sr = daily_sr * np.sqrt(252)

    # Higher statistical moments
    skew_val = float(skew(rets))
    kurt_val = float(kurtosis(rets, fisher=True)) + 3.0  # Pearson Kurtosis

    # Expected maximum Sharpe ratio among N trials under the null hypothesis
    # Approximation via Euler-Mascheroni constant
    euler_mascheroni = 0.5772156649
    if num_trials > 1:
        z_n = (1.0 - euler_mascheroni) * norm.ppf(
            1.0 - 1.0 / num_trials
        ) + euler_mascheroni * norm.ppf(1.0 - 1.0 / (num_trials * np.e))
        expected_max_sr = max(benchmark_sharpe, float(z_n))
    else:
        expected_max_sr = benchmark_sharpe

    # Asymptotic variance of the Sharpe ratio under non-normality
    sr_var = (
        1.0 - skew_val * daily_sr + ((kurt_val - 1.0) / 4.0) * (daily_sr**2)
    ) / max(T - 1, 1)
    sr_std = np.sqrt(max(sr_var, 1e-9))

    # Deflated Sharpe Z-Score and Probability
    z_stat = (daily_sr - (expected_max_sr / np.sqrt(252))) / sr_std
    dsr_prob = float(norm.cdf(z_stat))

    return {
        "annualized_sharpe": round(float(ann_sr), 2),
        "expected_max_null_sharpe": round(float(expected_max_sr), 2),
        "return_skewness": round(float(skew_val), 3),
        "return_kurtosis": round(float(kurt_val), 3),
        "dsr_probability": round(float(dsr_prob), 4),
        "is_statistically_significant": bool(dsr_prob >= 0.95),  # 95% confidence
    }
