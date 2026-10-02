"""
Unit tests for Split Conformal Prediction Intervals (Option 3).
"""

import numpy as np
import pytest
from src.conformal_prediction import (
    compute_conformal_quantile,
    calibrate_conformal_residuals_from_history,
    calculate_conformal_prediction_interval,
)


def test_compute_conformal_quantile_finite_sample():
    # Calibration residuals
    res = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    q90 = compute_conformal_quantile(res, alpha=0.10)
    # Romano / Angelopoulos ceiling((10+1)*0.9)/10 = 10/10 -> quantile level 1.0 -> 0.10
    assert q90 >= 0.09
    assert q90 <= 0.10


def test_conformal_interval_guarantee():
    # 100 sample residuals with 1.5% median
    residuals = np.abs(np.random.normal(0, 0.015, 100))
    interval = calculate_conformal_prediction_interval(
        ticker="NVDA",
        current_price=120.0,
        predicted_return_pct=1.0,  # +1%
        alpha=0.10,
        calibration_residuals=residuals,
    )

    assert interval["coverage_guarantee_pct"] == 90.0
    assert interval["price_interval_dollars"]["lower"] < 120.0 * 1.01
    assert interval["price_interval_dollars"]["upper"] > 120.0 * 1.01
    assert interval["interval_width_pct"] > 0
    assert interval["calibration_sample_size"] == 100


def test_conformal_interval_positive_alpha_detection():
    # Very small residuals with strong positive forecast
    small_residuals = np.array([0.002, 0.003, 0.004, 0.005, 0.006] * 20)
    interval = calculate_conformal_prediction_interval(
        ticker="AMZN",
        current_price=200.0,
        predicted_return_pct=3.0,  # +3% forecast
        alpha=0.10,
        calibration_residuals=small_residuals,
    )

    assert interval["statistical_edge"] == "GUARANTEED_POSITIVE_ALPHA"
    assert interval["return_interval_pct"]["lower"] > 0.0
    assert interval["price_interval_dollars"]["lower"] > 200.0
