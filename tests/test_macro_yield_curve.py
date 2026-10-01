"""
Unit tests for Nelson-Siegel-Svensson Yield Curve Decomposer (Sprint 2, Module 2.6)
"""

import pytest
import numpy as np
from src.macro_yield_curve import (
    svensson_yield,
    svensson_forward_rate,
    fit_svensson_curve,
    evaluate_treasury_yield_curve,
)


def test_svensson_yield_evaluation():
    m = np.array([1.0, 5.0, 10.0, 30.0])
    y = svensson_yield(
        m, beta0=4.5, beta1=-1.0, beta2=0.5, beta3=0.2, tau1=1.5, tau2=5.0
    )
    assert len(y) == 4
    assert (y > 0).all()
    assert (y < 15.0).all()


def test_svensson_forward_rate():
    m = np.array([1.0, 5.0, 10.0])
    f = svensson_forward_rate(
        m, beta0=4.5, beta1=-1.0, beta2=0.5, beta3=0.2, tau1=1.5, tau2=5.0
    )
    assert len(f) == 3
    assert (f > 0).all()


def test_fit_svensson_curve():
    mats = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
    yields = [5.20, 4.80, 4.40, 4.20, 4.30, 4.60]  # Classic inverted curve
    res = fit_svensson_curve(mats, yields)

    assert res["status"] == "SUCCESS"
    assert "parameters" in res
    assert "spread_10y_minus_2y" in res
    assert res["fitting_rmse"] < 0.25
    assert res["is_curve_inverted"] is True
    assert "INVERTED" in res["curve_shape"]


def test_evaluate_treasury_yield_curve_smoke():
    res = evaluate_treasury_yield_curve()
    assert res["status"] == "SUCCESS"
    assert "benchmark_curve" in res
    assert "10Y" in res["benchmark_curve"]
    assert "cro_advice" in res
