"""
Unit tests for Synthetic Crash Diffusion Engine (Sprint 3, Module 3.5)
"""

import pytest
import numpy as np
from src.stress_simulator import (
    SyntheticCrashDiffusionEngine,
    run_portfolio_stress_test,
)


def test_generate_synthetic_crash_paths():
    engine = SyntheticCrashDiffusionEngine(horizon_days=15, diffusion_steps=20)
    paths = engine.generate_synthetic_crash_paths(
        spot_price=100.0,
        scenario="COVID_2020_FLASH_CRASH",
        n_paths=10,
    )
    assert paths.shape == (10, 15)
    # Spot is 100, COVID crash scenario should lead to drawdown (< 100)
    assert (paths[:, -1] < 100.0).all()
    assert (paths > 0).all()


def test_stress_test_portfolio():
    res = run_portfolio_stress_test(equity=100000.0)
    assert res["status"] == "SUCCESS"
    assert "var_99_pct" in res
    assert "cvar_99_expected_shortfall_pct" in res
    assert res["var_99_dollars"] > 0
    assert len(res["historical_crisis_scenarios"]) == 4
    assert "COVID_2020_FLASH_CRASH" in res["historical_crisis_scenarios"]
