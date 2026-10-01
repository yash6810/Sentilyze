"""
Unit tests for Portfolio Optimizer & Ledoit-Wolf RMT Noise Filter (Sprint 1, Module 1.9)
"""

import pytest
import numpy as np
import pandas as pd

from src.portfolio_optimizer import (
    compute_ledoit_wolf_covariance,
    marchenko_pastur_bounds,
    denoise_covariance_rmt,
    compute_minimum_variance_portfolio,
    optimize_universe_portfolio,
)


def test_marchenko_pastur_bounds():
    # T=200, N=20 -> gamma = 0.1
    l_min, l_max = marchenko_pastur_bounds(T=200, N=20, sigma2=1.0)
    assert 0.0 < l_min < 1.0
    assert l_max > 1.0
    assert l_max > l_min


def test_ledoit_wolf_covariance():
    np.random.seed(42)
    # 100 rows, 5 assets
    data = np.random.normal(0, 0.02, size=(100, 5))
    df = pd.DataFrame(data, columns=[f"A{i}" for i in range(5)])

    cov_lw, shrinkage = compute_ledoit_wolf_covariance(df)
    assert cov_lw.shape == (5, 5)
    assert 0.0 <= shrinkage <= 1.0
    # Symmetric
    assert np.allclose(cov_lw, cov_lw.T)


def test_denoise_covariance_rmt():
    np.random.seed(42)
    # Highly correlated + noise
    data = np.random.normal(0, 0.02, size=(150, 10))
    # Add common factor to create strong signal
    factor = np.random.normal(0, 0.05, size=(150, 1))
    data += factor
    df = pd.DataFrame(data)

    cov_lw, _ = compute_ledoit_wolf_covariance(df)
    cov_denoised, meta = denoise_covariance_rmt(cov_lw, T=150)

    assert cov_denoised.shape == (10, 10)
    assert meta["noise_modes_shrunk"] > 0
    assert meta["signal_modes"] >= 1
    # Positive semi-definite
    evals = np.linalg.eigvalsh(cov_denoised)
    assert (evals >= -1e-5).all()


def test_compute_minimum_variance_portfolio():
    cov = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ]
    )
    w = compute_minimum_variance_portfolio(cov)
    assert len(w) == 2
    assert np.isclose(np.sum(w), 1.0)
    # Less volatile asset (index 0) should get higher weight
    assert w[0] > w[1]


def test_optimize_universe_portfolio_smoke():
    res = optimize_universe_portfolio(["NVDA", "AAPL"])
    assert "status" in res
    assert "optimized_weights" in res


def test_optimize_cdar_portfolio():
    from src.portfolio_optimizer import optimize_cdar_portfolio

    np.random.seed(42)
    # Generate 100 days of returns for 3 assets
    rets = np.random.normal(0.001, 0.02, size=(100, 3))
    df = pd.DataFrame(rets, columns=["A", "B", "C"])

    res = optimize_cdar_portfolio(df, alpha=0.95)
    assert res["status"] in ["SUCCESS", "APPROX_CONVERGED"]
    assert "weights" in res
    weights = list(res["weights"].values())
    assert pytest.approx(sum(weights), abs=1e-3) == 1.0
    assert res["cdar_drawdown_pct"] >= 0.0


def test_optimize_dro_portfolio():
    from src.portfolio_optimizer import optimize_dro_portfolio

    np.random.seed(42)
    rets = np.random.normal(0.0005, 0.015, size=(80, 4))
    df = pd.DataFrame(rets, columns=["A", "B", "C", "D"])

    res = optimize_dro_portfolio(df, epsilon=0.05, risk_aversion=1.0)
    assert res["status"] in ["SUCCESS", "APPROX_CONVERGED"]
    weights = list(res["weights"].values())
    assert pytest.approx(sum(weights), abs=1e-3) == 1.0


def test_compute_conformal_black_litterman():
    from src.portfolio_optimizer import compute_conformal_black_litterman

    cov = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.01],
            [0.005, 0.01, 0.06],
        ]
    )
    w_mkt = np.array([0.5, 0.3, 0.2])
    views = {0: 0.15}  # Strong bullish view on asset 0 (+15%)
    # Narrow conformal interval (high certainty)
    conformal_intervals = {0: (0.12, 0.18)}

    res = compute_conformal_black_litterman(
        cov_matrix=cov,
        market_weights=w_mkt,
        views=views,
        conformal_intervals=conformal_intervals,
        tau=0.05,
    )

    assert res["status"] == "SUCCESS"
    assert len(res["bl_weights"]) == 3
    assert pytest.approx(sum(res["bl_weights"]), abs=1e-3) == 1.0
    # Bullish view on asset 0 should expand its weight beyond 0.50
    assert res["bl_weights"][0] > 0.50


def test_optimize_maximum_diversification_ratio():
    from src.portfolio_optimizer import optimize_maximum_diversification_ratio

    # 3 assets with varying volatilities and low correlation
    cov = np.array(
        [
            [0.04, 0.001, 0.001],
            [0.001, 0.09, 0.001],
            [0.001, 0.001, 0.16],
        ]
    )

    res = optimize_maximum_diversification_ratio(cov)
    assert res["status"] in ["SUCCESS", "APPROX_CONVERGED"]
    assert len(res["weights"]) == 3
    assert pytest.approx(sum(res["weights"]), abs=1e-3) == 1.0
    # Diversification ratio should be strictly greater than 1.0 for non-perfectly correlated assets
    assert res["diversification_ratio"] > 1.0


def test_solve_slippage_socp_rebalancing():
    from src.portfolio_optimizer import solve_slippage_socp_rebalancing

    w0 = np.array([0.5, 0.5])
    mu = np.array([0.08, 0.04])
    cov = np.array([[0.04, 0.01], [0.01, 0.04]])

    res = solve_slippage_socp_rebalancing(
        current_weights=w0,
        expected_returns=mu,
        cov_matrix=cov,
        linear_fee_bps=10.0,
        quadratic_slippage_coeff=0.10,
        max_turnover=0.30,
    )

    assert res["status"] in ["SUCCESS", "APPROX_CONVERGED"]
    assert pytest.approx(sum(res["optimal_weights"]), abs=1e-3) == 1.0
    assert res["portfolio_turnover"] <= 0.3001
    assert res["total_transaction_drag_bps"] >= 0.0
