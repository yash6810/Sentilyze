from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.data_ingestion import get_price_history

logger = get_logger(__name__)


def calculate_portfolio_diversity_grade(
    tickers: List[str],
    period: str = "6mo",
    custom_returns: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    cleaned_tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]
    cleaned_tickers = list(dict.fromkeys(cleaned_tickers))

    if len(cleaned_tickers) == 0:
        return {
            "grade": "N/A",
            "grade_color": "#94A3B8",
            "average_correlation": 0.0,
            "effective_bets": 0.0,
            "max_correlated_pair": None,
            "min_correlated_pair": None,
            "correlation_matrix": {},
            "tickers": [],
            "status": "EMPTY_PORTFOLIO",
            "message": "No active holdings provided to evaluate diversity.",
            "diagnostics": [],
        }

    if len(cleaned_tickers) == 1:
        single_t = cleaned_tickers[0]
        return {
            "grade": "D",
            "grade_color": "#EF4444",
            "average_correlation": 1.0,
            "effective_bets": 1.0,
            "max_correlated_pair": (single_t, single_t, 1.0),
            "min_correlated_pair": (single_t, single_t, 1.0),
            "correlation_matrix": {single_t: {single_t: 1.0}},
            "tickers": [single_t],
            "status": "SINGLE_ASSET_CONCENTRATION",
            "message": f"Portfolio contains only 1 asset ({single_t}). Maximum concentration risk.",
            "diagnostics": [
                "Warning: Single asset holding has 100% idiosyncratic risk.",
                "Recommendation: Add at least 3-5 uncorrelated assets across different sectors.",
            ],
        }

    if custom_returns is not None and not custom_returns.empty:
        returns_df = custom_returns[
            [t for t in cleaned_tickers if t in custom_returns.columns]
        ]
    else:
        price_series = {}
        for t in cleaned_tickers:
            try:
                df = get_price_history(t, period=period, use_cache=True)
                if not df.empty and "Close" in df.columns:
                    price_series[t] = df["Close"]
            except Exception as e:
                logger.debug(f"Could not load price history for {t}: {e}")

        if not price_series:
            return {
                "grade": "N/A",
                "grade_color": "#94A3B8",
                "average_correlation": 0.0,
                "effective_bets": 0.0,
                "max_correlated_pair": None,
                "min_correlated_pair": None,
                "correlation_matrix": {},
                "tickers": cleaned_tickers,
                "status": "DATA_UNAVAILABLE",
                "message": "Insufficient price data to compute return correlations.",
                "diagnostics": [],
            }

        prices_df = pd.DataFrame(price_series).dropna(how="all")
        returns_df = prices_df.pct_change().dropna(how="all")

    valid_tickers = [t for t in cleaned_tickers if t in returns_df.columns]
    if len(valid_tickers) < 2:
        return {
            "grade": "D",
            "grade_color": "#EF4444",
            "average_correlation": 1.0,
            "effective_bets": 1.0,
            "max_correlated_pair": None,
            "min_correlated_pair": None,
            "correlation_matrix": {},
            "tickers": valid_tickers,
            "status": "INSUFFICIENT_OVERLAPPING_DATA",
            "message": "Fewer than 2 assets have valid overlapping return history.",
            "diagnostics": [],
        }

    returns_df = returns_df[valid_tickers].dropna()
    if len(returns_df) < 10:
        corr_matrix = pd.DataFrame(
            np.eye(len(valid_tickers)), index=valid_tickers, columns=valid_tickers
        )
    else:
        corr_matrix = returns_df.corr(method="pearson").fillna(0.0)

    n = len(valid_tickers)
    off_diag_corrs = []
    pairs_list = []

    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = valid_tickers[i], valid_tickers[j]
            r_val = float(corr_matrix.loc[t1, t2])
            off_diag_corrs.append(r_val)
            pairs_list.append((t1, t2, r_val))

    avg_corr = float(np.mean(off_diag_corrs)) if off_diag_corrs else 0.0
    pairs_list.sort(key=lambda x: x[2], reverse=True)
    max_pair = pairs_list[0] if pairs_list else None
    min_pair = pairs_list[-1] if pairs_list else None

    try:
        eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        p_weights = eigenvalues / np.sum(eigenvalues)
        shannon_entropy = -np.sum(p_weights * np.log(p_weights))
        effective_bets = float(np.exp(shannon_entropy))
    except Exception:
        effective_bets = float(n / (1.0 + (n - 1) * max(0.0, avg_corr)))

    effective_bets = min(max(1.0, effective_bets), float(n))

    if avg_corr < 0.15:
        grade = "A+"
        grade_color = "#10B981"
        grade_desc = "Elite Uncorrelated Risk Parity"
    elif avg_corr < 0.25:
        grade = "A-"
        grade_color = "#34D399"
        grade_desc = "Strong Sector Diversification"
    elif avg_corr < 0.40:
        grade = "B+"
        grade_color = "#38BDF8"
        grade_desc = "Healthy Institutional Balance"
    elif avg_corr < 0.55:
        grade = "B-"
        grade_color = "#FBBF24"
        grade_desc = "Moderate Cluster Correlation"
    elif avg_corr < 0.70:
        grade = "C"
        grade_color = "#FB923C"
        grade_desc = "Elevated Systemic Covariance"
    else:
        grade = "D"
        grade_color = "#EF4444"
        grade_desc = "Severe Directional Concentration"

    diagnostics = []
    if max_pair and max_pair[2] > 0.75:
        diagnostics.append(
            f"Warning: High Cluster Overlap between {max_pair[0]} and {max_pair[1]} ({max_pair[2]:.2f} correlation). Consider sizing down to avoid duplicate exposure."
        )
    if min_pair and min_pair[2] < 0.05:
        diagnostics.append(
            f"Protection: Low Correlation Hedge between {min_pair[0]} and {min_pair[1]} ({min_pair[2]:+.2f} corr) provides natural volatility dampening."
        )
    if effective_bets < (n * 0.5):
        diagnostics.append(
            f"Attention: Effective Bets ({effective_bets:.1f}) is less than half of nominal holdings ({n}). Add defensive sectors to expand breadth."
        )
    else:
        diagnostics.append(
            f"Strength: Strong Statistical Breadth — Portfolio behaves like {effective_bets:.1f} independent profit engines across market regimes."
        )

    return {
        "grade": grade,
        "grade_color": grade_color,
        "grade_description": grade_desc,
        "average_correlation": round(avg_corr, 3),
        "effective_bets": round(effective_bets, 2),
        "nominal_holdings": n,
        "max_correlated_pair": max_pair,
        "min_correlated_pair": min_pair,
        "correlation_matrix": corr_matrix.round(3).to_dict(),
        "corr_matrix_df": corr_matrix.round(3),
        "tickers": valid_tickers,
        "status": "SUCCESS",
        "diagnostics": diagnostics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
