"""
Feature & Factor Neutralization for Quantitative Machine Learning.

Orthogonalizes feature matrices and model predictions against market beta (SPY/QQQ)
and sector exposures using Moore-Penrose pseudo-inverse projection.
Reference: Numerai & AQR Factor Neutralization Standards.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def neutralize_features(
    df: pd.DataFrame,
    target_columns: list,
    factor_columns: list,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """
    Neutralizes target feature columns with respect to factor columns (e.g. SPY return, sector beta).

    Formula:
        X_neutral = X - proportion * F * (F^T F)^(-1) * F^T * X
    """
    if df.empty or not factor_columns or not target_columns:
        return df

    df_out = df.copy()

    # Align and extract matrices
    X = df_out[target_columns].values.astype(np.float64)
    F = df_out[factor_columns].values.astype(np.float64)

    # Add intercept/bias column to factors if not present
    if not np.allclose(F[:, 0], 1.0):
        F = np.column_stack([np.ones(len(F)), F])

    try:
        # Compute pseudo-inverse projection: pinv(F) @ X
        pinv_F = np.linalg.pinv(F)
        exposure = F @ (pinv_F @ X)
        X_neutral = X - (proportion * exposure)
        df_out[target_columns] = X_neutral
    except Exception as e:
        logger.warning(
            f"Feature neutralization failed: {e}. Returning original features."
        )

    return df_out


def neutralize_predictions(
    predictions: Union[pd.Series, np.ndarray],
    factor_matrix: Union[pd.DataFrame, np.ndarray],
    proportion: float = 1.0,
) -> np.ndarray:
    """
    Removes linear factor exposure from raw prediction scores.
    """
    if isinstance(predictions, pd.Series):
        preds = predictions.values.reshape(-1, 1).astype(np.float64)
    else:
        preds = np.asarray(predictions, dtype=np.float64).reshape(-1, 1)

    if isinstance(factor_matrix, pd.DataFrame):
        F = factor_matrix.values.astype(np.float64)
    else:
        F = np.asarray(factor_matrix, dtype=np.float64)

    if F.ndim == 1:
        F = F.reshape(-1, 1)

    # Add intercept
    F_aug = np.column_stack([np.ones(len(F)), F])

    try:
        pinv_F = np.linalg.pinv(F_aug)
        linear_exposure = F_aug @ (pinv_F @ preds)
        neutral_preds = preds - (proportion * linear_exposure)
        return neutral_preds.ravel()
    except Exception:
        return preds.ravel()
