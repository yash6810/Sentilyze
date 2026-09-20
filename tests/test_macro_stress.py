import numpy as np
import pytest
from src.risk_evt import (
    fit_gpd_peaks_over_threshold,
    compute_evt_var_cvar,
)
from src.macro_stress import (
    CRISIS_SCENARIOS,
    simulate_crisis_replay,
    run_full_crisis_stress_test,
)


def test_gpd_fit():
    np.random.seed(42)
    # Generate Student-t distributed returns with fat tails (df=3)
    returns = np.random.standard_t(df=3, size=500) * 0.015

    fit = fit_gpd_peaks_over_threshold(returns, threshold_quantile=0.90)
    assert fit["status"] in ["FIT_SUCCESS", "FEW_EXCEEDANCES"]
    assert fit["threshold_u"] > 0.0
    assert fit["scale_beta"] > 0.0
    assert -0.5 < fit["shape_xi"] < 0.5


def test_evt_var_and_cvar():
    np.random.seed(42)
    returns = np.random.standard_t(df=3, size=500) * 0.02

    res = compute_evt_var_cvar(returns, confidence_level=0.99)
    assert res["status"] == "CALCULATION_SUCCESS"
    assert res["evt_var_pct"] > 0.0
    # Expected Shortfall (CVaR) must always be >= VaR
    assert res["evt_cvar_pct"] >= res["evt_var_pct"]
    assert res["tail_classification"] in [
        "EXTREME_FAT_TAIL",
        "MODERATE_FAT_TAIL",
        "NEAR_GAUSSIAN",
    ]


def test_crisis_replay_simulation():
    mock_portfolio = {
        "total_equity": 100_000.0,
        "cash": 40_000.0,  # 40% cash cushion
        "open_positions": {
            "NVDA": {"market_value": 30_000.0},
            "CAT": {"market_value": 30_000.0},
        },
    }

    res_covid = simulate_crisis_replay(
        "2020_COVID_FLASH_CRASH", portfolio_data=mock_portfolio
    )
    assert res_covid["scenario_key"] == "2020_COVID_FLASH_CRASH"
    assert res_covid["initial_total_equity"] == 100_000.0
    assert res_covid["cash_buffer_dollars"] == 40_000.0

    # Drawdown on portfolio must be less severe than unhedged benchmark because of 40% cash
    assert abs(res_covid["portfolio_drawdown_pct"]) < abs(
        res_covid["unhedged_market_drawdown_pct"]
    )
    assert res_covid["cash_cushion_protection_pct"] > 0.0
    assert res_covid["simulated_final_equity"] > 40_000.0


def test_run_full_crisis_stress_test():
    res = run_full_crisis_stress_test()
    assert res["status"] == "STRESS_TEST_COMPLETE"
    assert res["total_scenarios_replayed"] == 4
    assert "2008_LEHMAN_GFC" in res["scenarios"]
    assert "1987_BLACK_MONDAY" in res["scenarios"]
    assert "worst_case_scenario" in res
