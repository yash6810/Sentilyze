"""
Fixed-Width Window Fractional Differentiation (FFD) for Financial Time Series.

Implements the memory-preserving stationarity transformation based on:
Marcos Lopez de Prado (2018), 'Advances in Financial Machine Learning', Chapter 5.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_weights_ffd(
    d: float, threshold: float = 1e-4, max_lags: int = 2000
) -> np.ndarray:
    """
    Generate weights for Fixed-Width Window Fractional Differentiation.

    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k
    """
    weights = [1.0]
    k = 1
    while k < max_lags:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights[::-1])


def fractional_differentiation_ffd(
    series: pd.Series,
    d: float = 0.4,
    threshold: float = 1e-4,
) -> pd.Series:
    """
    Applies Fixed-Width Window Fractional Differentiation to a price series.
    """
    if series.empty:
        return series

    weights = get_weights_ffd(d, threshold=threshold)
    width = len(weights)

    values = series.values
    n = len(values)

    if n < width:
        logger.warning(
            f"Series length ({n}) < FFD window width ({width}). Using fallback difference."
        )
        return series.diff().dropna()

    res = np.empty(n - width + 1)
    for i in range(width - 1, n):
        res[i - width + 1] = np.dot(weights, values[i - width + 1 : i + 1])

    out_index = series.index[width - 1 :]
    name = f"{series.name}_fracdiff_{d:.2f}" if series.name else f"fracdiff_{d:.2f}"
    return pd.Series(res, index=out_index, name=name)


def find_optimal_d(
    series: pd.Series,
    d_range: Optional[np.ndarray] = None,
    adf_pvalue_threshold: float = 0.05,
) -> Tuple[float, pd.Series]:
    """
    Finds the minimum fractional differentiation order d that passes the Augmented Dickey-Fuller test.
    """
    if d_range is None:
        d_range = np.linspace(0.1, 0.9, 9)

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        optimal_d = 0.40
        return optimal_d, fractional_differentiation_ffd(series, d=optimal_d)

    clean_series = series.dropna()
    if len(clean_series) < 30:
        return 0.40, fractional_differentiation_ffd(series, d=0.40)

    best_d = 0.50
    best_series = None

    for d in d_range:
        ffd_s = fractional_differentiation_ffd(clean_series, d=d)
        if len(ffd_s) < 20:
            continue
        try:
            adf_stat, p_val, _, _, _, _ = adfuller(ffd_s.values, autolag="AIC")
            if p_val < adf_pvalue_threshold:
                best_d = float(d)
                best_series = ffd_s
                break
        except Exception:
            continue

    if best_series is None:
        best_d = 0.40
        best_series = fractional_differentiation_ffd(clean_series, d=best_d)

    return best_d, best_series
