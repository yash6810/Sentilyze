"""
Lead-Lag Granger Causality & Supply Chain Price Discovery Engine for Sentilyze.
Pillar 4 Advanced Module:
- Computes pairwise Granger Causality tests across equity universe assets.
- Uses high-performance vector OLS and Fisher F-test p-value evaluation.
- Detects lead-lag relationships (e.g., TSM leading NVDA, NVDA leading AMD).
- Ranks market leaders by predictive out-degree centrality.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.utils import get_logger

logger = get_logger(__name__)


def _granger_f_test(
    y: np.ndarray, x: np.ndarray, lag: int = 2
) -> float:
    """
    Computes pure vector OLS Granger causality F-test p-value (testing if x Granger-causes y).
    """
    n = len(y)
    if n <= 2 * lag + 2:
        return 1.0

    # Build design matrices
    Y_target = y[lag:]

    # Restricted design matrix (lags of Y only)
    X_restr = [np.ones(n - lag)]
    for i in range(1, lag + 1):
        X_restr.append(y[lag - i: n - i])
    X_restr = np.column_stack(X_restr)

    # Unrestricted design matrix (lags of Y and lags of X)
    X_unrestr = list(X_restr.T)
    for j in range(1, lag + 1):
        X_unrestr.append(x[lag - j: n - j])
    X_unrestr = np.column_stack(X_unrestr)

    # Solve OLS via least squares
    beta_r, res_r, _, _ = np.linalg.lstsq(X_restr, Y_target, rcond=None)
    beta_u, res_u, _, _ = np.linalg.lstsq(X_unrestr, Y_target, rcond=None)

    rss_r = np.sum((Y_target - X_restr @ beta_r) ** 2)
    rss_u = np.sum((Y_target - X_unrestr @ beta_u) ** 2)

    df_num = lag
    df_denom = n - lag - (2 * lag + 1)
    if df_denom <= 0 or rss_u <= 1e-12:
        return 1.0

    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_denom)
    if f_stat <= 0:
        return 1.0

    p_value = float(stats.f.sf(f_stat, df_num, df_denom))
    return float(np.clip(p_value, 0.0, 1.0))


def compute_lead_lag_matrix(
    price_series_dict: Dict[str, pd.Series], max_lag: int = 2
) -> pd.DataFrame:
    """
    Computes pairwise Granger Causality p-values across a dictionary of asset price series.
    A low p-value (p < 0.05) indicates that Row Asset Granger-causes / leads Column Asset.

    Args:
        price_series_dict: Dict of {ticker: close_price_series}
        max_lag: Maximum lag days to test

    Returns:
        Square DataFrame of p-values (Row leads Column).
    """
    tickers = list(price_series_dict.keys())
    matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)

    # Compute daily percentage log returns
    ret_dict = {}
    for t, s in price_series_dict.items():
        ret = np.log(s / s.shift(1)).dropna()
        ret_dict[t] = ret

    for leader in tickers:
        for follower in tickers:
            if leader == follower:
                matrix.loc[leader, follower] = 1.0
                continue

            s_leader = ret_dict[leader]
            s_follower = ret_dict[follower]

            # Common aligned dates
            aligned = pd.concat([s_follower, s_leader], axis=1).dropna()
            if len(aligned) < 20:
                matrix.loc[leader, follower] = 1.0
                continue

            y_arr = aligned.iloc[:, 0].values
            x_arr = aligned.iloc[:, 1].values

            p_val = _granger_f_test(y_arr, x_arr, lag=max_lag)
            matrix.loc[leader, follower] = round(p_val, 4)

    return matrix


def rank_market_price_leaders(
    lead_lag_matrix: pd.DataFrame, alpha_threshold: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Ranks stocks by their predictive influence (number of peers they statistically lead).

    Args:
        lead_lag_matrix: Granger Causality p-value matrix
        alpha_threshold: Significance level (default 0.05 for 95% confidence)

    Returns:
        List of ranked leaders with lead score, influenced followers, and significance count.
    """
    ranks = []
    for leader in lead_lag_matrix.index:
        row = lead_lag_matrix.loc[leader]
        significant_followers = [
            f for f in lead_lag_matrix.columns
            if f != leader and float(row[f]) < alpha_threshold
        ]

        ranks.append({
            "ticker": leader,
            "leads_count": len(significant_followers),
            "followers": significant_followers,
            "influence_score": round((len(significant_followers) / max(1, len(lead_lag_matrix.columns) - 1)) * 100, 1),
            "status": "👑 PRIMARY PRICE DISCOVERY LEADER" if len(significant_followers) >= 3 else "⚡ PEER DRIVER" if len(significant_followers) >= 1 else "⚪ LAGGER / FOLLOWER",
        })

    ranks.sort(key=lambda x: x["leads_count"], reverse=True)
    return ranks
