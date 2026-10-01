"""
Unit tests for Transient Market Impact Decay & Execution Algorithms (Sprint 2, Module 2.10)
"""

import pytest
import numpy as np
from src.execution_algo import (
    transient_impact_kernel,
    compute_optimal_cooldown,
    simulate_child_order_schedule,
    optimize_order_execution_trajectory,
)


def test_transient_impact_kernel():
    lags = np.array([0.0, 2.0, 10.0, 60.0])
    k = transient_impact_kernel(lags, gamma0=0.05, tau0=2.0, alpha=0.5)
    assert len(k) == 4
    # Monotonically decaying with lag
    assert k[0] > k[1] > k[2] > k[3]


def test_compute_optimal_cooldown():
    # Decay target 25% with tau0=2.0s and alpha=0.5
    # (0.25)^(-2) - 1 = 16 - 1 = 15 -> cooldown = 2.0 * 15 = 30.0s
    cooldown = compute_optimal_cooldown(
        decay_target_pct=0.25, tau0_seconds=2.0, alpha=0.5
    )
    assert cooldown == 30.0


def test_simulate_child_order_schedule():
    res = simulate_child_order_schedule(
        total_shares=5000,
        slice_count=5,
        interval_seconds=30.0,
        adv=500000.0,
        spot_price=100.0,
    )
    assert res["shares_per_slice"] == 1000.0
    assert len(res["impact_trajectory"]) == 5
    assert res["total_dollar_slippage"] > 0.0


def test_optimize_order_execution_trajectory_smoke():
    traj = optimize_order_execution_trajectory(total_shares=2000, urgency="HIGH")
    assert traj["status"] == "SUCCESS"
    assert "optimal_policy" in traj
    assert traj["optimal_policy"]["slice_count"] == 3


def test_decision_transformer_order_slicer():
    from src.execution_algo import (
        DecisionTransformerOrderSlicer,
        generate_decision_transformer_schedule,
    )
    import torch

    torch.manual_seed(42)
    model = DecisionTransformerOrderSlicer(state_dim=4, d_model=16, n_heads=2)

    res = generate_decision_transformer_schedule(
        total_shares=10000.0,
        horizon_steps=5,
        target_return_to_go=0.08,
        volatility=0.015,
        spread_bps=4.0,
        model=model,
    )

    assert res["status"] == "SUCCESS"
    assert len(res["slices"]) == 5
    assert pytest.approx(sum(res["slices"]), rel=1e-3) == 10000.0
    assert pytest.approx(res["terminal_fill_pct"], abs=1e-3) == 1.0
    assert res["remaining_shares"][-1] == 0.0
    assert len(res["rtg_trajectory"]) == 6
