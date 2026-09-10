"""
Unit Tests for All 33 Cross-Disciplinary & Frontier Quantitative Models
========================================================================
Tests mathematical correctness, boundary conditions, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from src.cross_disciplinary_alphas import (
    compute_lotka_volterra_ecology,
    compute_ising_market_phase,
    compute_navier_stokes_reynolds_number,
    compute_sir_narrative_r0,
    compute_feynman_path_least_action,
    compute_ant_colony_venue_allocation,
    compute_theta_gamma_phase_coupling,
    compute_shannon_alpha_capacity,
    compute_bayesian_dark_pool_equilibrium,
    compute_circadian_seasonal_risk_scalar,
    compute_omori_aftershock_rate,
    compute_epigenetic_factor_methylation,
    compute_gravitational_liquidity_pull,
    compute_quantum_resistance_tunneling,
    compute_tensile_fatigue_fracture,
    compute_doppler_order_flow_shift,
    compute_lyapunov_predictability_horizon,
    compute_lanchester_combat_power,
    parse_price_morphology_grammar,
    compute_carnot_profit_efficiency,
    compute_wave_superposition,
    compute_sandpile_criticality,
    compute_hawk_dove_equilibrium,
    compute_inflaton_bubble_decay,
    compute_pid_kalman_sizing,
    compute_options_gex_regime,
    compute_symbolic_genetic_alpha,
    compute_wyckoff_fvg_score,
    compute_vocal_stress_sentiment,
    compute_lead_lag_spillover_signal,
    compute_self_play_equilibrium_weight,
    compute_tda_betti_cavity_score,
    compute_conformal_safety_bands,
)


def test_lotka_volterra_ecology():
    r_vol = pd.Series([100.0, 150.0, 200.0])
    i_flow = pd.Series([50.0, 60.0, 70.0])
    res = compute_lotka_volterra_ecology(r_vol, i_flow)
    assert "prey_density" in res
    assert "predator_density" in res
    assert res["overgrazing_ratio"] > 0


def test_ising_market_phase():
    spins = np.array([1, 1, 1, 1, -1, 1])
    res = compute_ising_market_phase(spins, coupling_strength=0.8, temperature=0.5)
    assert "magnetization" in res
    assert res["phase"] in ["FERROMAGNETIC_STAMPEDE", "PARAMAGNETIC_ORDERED"]


def test_navier_stokes_reynolds_number():
    res = compute_navier_stokes_reynolds_number(
        order_flow_velocity=500.0, spread_length_scale=0.05, depth_viscosity=0.001
    )
    assert res["reynolds_number"] > 0
    assert "flow_regime" in res


def test_sir_narrative_r0():
    ts = pd.Series([10, 20, 45, 90, 180])
    res = compute_sir_narrative_r0(ts, recovery_rate=0.10)
    assert res["reproduction_number_r0"] > 1.0
    assert res["is_viral_growth"] is True


def test_feynman_path_least_action():
    p1 = np.array([100, 101, 102, 103])
    p2 = np.array([100, 120, 80, 103])
    res = compute_feynman_path_least_action([p1, p2])
    assert res["best_path_index"] == 0  # Smooth path has less action than erratic path


def test_ant_colony_venue_allocation():
    fills = {"IEX": 0.95, "NASDAQ": 0.80, "DARK_POOL": 0.90}
    slips = {"IEX": 0.0005, "NASDAQ": 0.002, "DARK_POOL": 0.0008}
    weights = compute_ant_colony_venue_allocation(fills, slips)
    assert round(sum(weights.values()), 2) == 1.0
    assert weights["IEX"] > weights["NASDAQ"]


def test_theta_gamma_phase_coupling():
    m_prices = pd.Series(np.linspace(100, 120, 30))
    h_prices = pd.Series(np.linspace(115, 120, 10))
    res = compute_theta_gamma_phase_coupling(m_prices, h_prices)
    assert res["is_phase_aligned"] is True
    assert res["gating_multiplier"] >= 1.0


def test_shannon_alpha_capacity():
    sig = pd.Series(np.random.normal(0, 1, 100))
    noise = pd.Series(np.random.normal(0, 0.2, 100))
    res = compute_shannon_alpha_capacity(sig, noise)
    assert res["shannon_capacity_bits_per_year"] > 0


def test_bayesian_dark_pool_equilibrium():
    res = compute_bayesian_dark_pool_equilibrium(hidden_block_size=10000)
    assert "optimal_strategy" in res
    assert res["recommended_slice_pct"] > 0


def test_circadian_seasonal_risk_scalar():
    t = pd.Timestamp("2026-04-10 10:00:00")
    res = compute_circadian_seasonal_risk_scalar(t)
    assert res["circadian_risk_scalar"] > 0


def test_remaining_alphas_sanity():
    # 11-33 Rapid Sanity Tests
    assert compute_omori_aftershock_rate(5.0)["aftershock_intensity_rate"] > 0
    assert (
        len(
            compute_epigenetic_factor_methylation(30.0, {"momentum": 0.5, "value": 0.5})
        )
        == 2
    )
    assert (
        compute_gravitational_liquidity_pull(100.0, 105.0, 50000)["gravitational_force"]
        > 0
    )
    assert (
        0.0
        <= compute_quantum_resistance_tunneling(50.0, 60.0)["tunneling_probability"]
        <= 1.0
    )
    assert (
        compute_tensile_fatigue_fracture(4, 20)["fracture_breakout_probability"] > 0.5
    )
    assert compute_doppler_order_flow_shift(5.0, 15.0)["frequency_shift_ratio"] == 3.0
    assert (
        compute_lyapunov_predictability_horizon(pd.Series(np.random.normal(0, 1, 50)))[
            "predictability_horizon_days"
        ]
        > 0
    )
    assert compute_lanchester_combat_power(2000, 1000)["combat_power_ratio"] == 4.0
    assert (
        parse_price_morphology_grammar(["CONSOLIDATE", "SPRING", "MARKUP"])[
            "is_grammatically_valid_accumulation"
        ]
        is True
    )
    assert compute_carnot_profit_efficiency(0.03, 0.01)["carnot_efficiency"] > 0
    assert (
        compute_wave_superposition(1, 1, 1, 1)["constructive_amplitude_multiplier"]
        == 2.0
    )
    assert compute_sandpile_criticality(1.5, 2.0)["sandpile_criticality_index"] == 3.0
    assert (
        compute_hawk_dove_equilibrium(2.5, 0.02)["liquidity_provider_pullout"] is True
    )
    assert compute_inflaton_bubble_decay(2.0, 1.0)["bubble_exhaustion_top"] is True
    assert (
        0.25
        <= compute_pid_kalman_sizing(0.1, 0.05, 0.01)["damped_position_size"]
        <= 1.75
    )
    assert compute_options_gex_regime(500.0)["gamma_regime"] == "LONG_GAMMA_MEAN_REVERT"
    assert (
        len(
            compute_symbolic_genetic_alpha(
                pd.Series([1, 2]), pd.Series([1.5, 1.2]), pd.Series([0.1, -0.1])
            )
        )
        == 2
    )
    assert compute_wyckoff_fvg_score(True, True, True)["high_conviction_setup"] is True
    assert (
        compute_vocal_stress_sentiment(1.5, 2.0)["executive_hesitation_detected"]
        is True
    )
    assert (
        compute_lead_lag_spillover_signal(3.0)["expected_downstream_spillover_pct"] > 0
    )
    assert compute_self_play_equilibrium_weight(1.0, 0.2) == 0.8
    assert (
        compute_tda_betti_cavity_score(np.eye(5))["topological_cavity_detected"]
        is False
    )
    assert compute_conformal_safety_bands(150.0, 2.5)["conformal_lower_bound"] < 150.0
