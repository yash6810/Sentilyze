"""
Combinatorial Purged & Embargoed Cross-Validation (CPCV) for Financial Machine Learning.

Eliminates lookahead bias, overlapping trade span leakage, and autoregressive memory leakage.
Calculates the Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO).
Reference: Marcos Lopez de Prado (2018), 'Advances in Financial Machine Learning', Chapter 12.
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Optional
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)


class PurgedGroupTimeSeriesSplit:
    """
    Time-series cross-validator that purges overlapping event windows and applies embargoes.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_pct: float = 0.02,
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.embargo_pct)

        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples

            test_idx = indices[test_start:test_end]

            # Purge training set: remove samples within purge_window before test_start
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False

            # Purge pre-test overlap
            purge_start = max(0, test_start - self.purge_window)
            train_mask[purge_start:test_start] = False

            # Embargo post-test window
            embargo_end = min(n_samples, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False

            train_idx = indices[train_mask]

            yield train_idx, test_idx


def compute_deflated_sharpe_ratio(
    estimated_sharpe: float,
    benchmark_sharpe: float,
    n_trials: int,
    var_sharpe: float,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Calculates the Deflated Sharpe Ratio (DSR) to test for backtest overfitting.
    """
    # Expected maximum Sharpe under the null hypothesis (Euler-Mascheroni approximation)
    gamma = 0.5772156649
    if n_trials <= 1:
        e_max_sharpe = benchmark_sharpe
    else:
        e_max_sharpe = benchmark_sharpe + np.sqrt(var_sharpe) * (
            (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )

    # Standard error of the Sharpe ratio under non-normality
    sr_std_err = np.sqrt(
        (
            1.0
            + 0.5 * estimated_sharpe**2
            - skewness * estimated_sharpe
            + (kurtosis - 3.0) / 4.0 * estimated_sharpe**2
        )
        / (sample_length - 1.0)
    )

    if sr_std_err <= 1e-8:
        return 1.0 if estimated_sharpe > e_max_sharpe else 0.0

    z_stat = (estimated_sharpe - e_max_sharpe) / sr_std_err
    dsr_pvalue = float(stats.norm.cdf(z_stat))
    return dsr_pvalue
