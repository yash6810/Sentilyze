"""
Unit tests for Mahalanobis Anomaly Sentinel (Sprint 1, Module 1.8)
"""

import pytest
import numpy as np
import pandas as pd
from src.risk_guardian import MahalanobisAnomalySentinel, evaluate_mahalanobis_sentinel


def test_fit_and_compute_distance():
    sentinel = MahalanobisAnomalySentinel()
    np.random.seed(42)
    # Generate 100 samples of 5D Gaussian data
    X = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0], cov=np.eye(5), size=100)
    df_feats = pd.DataFrame(X, columns=sentinel.FEATURE_COLUMNS)
    mu, inv_cov = sentinel.fit_reference_distribution(df_feats)

    # In-distribution point near origin should have small distance
    x_in = np.array([0.1, -0.1, 0.05, 0.0, -0.05])
    d_in = sentinel.compute_mahalanobis_distance(x_in, mu, inv_cov)
    assert d_in < 2.0

    # Extreme outlier point should have high distance
    x_out = np.array([8.0, 9.0, -10.0, 12.0, 7.0])
    d_out = sentinel.compute_mahalanobis_distance(x_out, mu, inv_cov)
    assert d_out > 10.0


def test_audit_asset_smoke():
    res = evaluate_mahalanobis_sentinel("NVDA")
    assert "status" in res
    assert "is_anomaly" in res
    assert "regime_verdict" in res
    assert "mahalanobis_distance" in res
