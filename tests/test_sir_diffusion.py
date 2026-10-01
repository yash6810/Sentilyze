"""
Unit tests for Meme Viral SIR Differential Diffusion Model (Sprint 2, Module 2.8)
"""

import pytest
from src.sentiment_analysis import MemeSIRDiffusionModel, fit_sir_meme_diffusion


def test_simulate_diffusion():
    model = MemeSIRDiffusionModel(beta=0.5, gamma=0.2)
    assert model.r0 == 2.5

    df_sim = model.simulate_diffusion(days=15, dt=0.5)
    assert not df_sim.empty
    assert "susceptible_ratio" in df_sim.columns
    assert "infected_spreader_ratio" in df_sim.columns
    # S(t) should decrease monotonically
    assert df_sim["susceptible_ratio"].iloc[-1] < df_sim["susceptible_ratio"].iloc[0]


def test_evaluate_social_saturation_peak():
    model = MemeSIRDiffusionModel(beta=0.45, gamma=0.15)
    # 200 mentions vs 20 baseline (10x surge)
    res = model.evaluate_social_saturation(
        current_mentions=200.0,
        baseline_mentions=20.0,
        mention_velocity_pct=5.0,
    )
    assert res["is_hype_saturated"] is True
    assert "EXHAUSTION" in res["diffusion_regime"]
    assert res["cro_recommendation"] == "TRIM_LONGS_TIGHTEN_STOPS"


def test_fit_sir_meme_diffusion_smoke():
    res = fit_sir_meme_diffusion()
    assert "reproduction_number_R0" in res
    assert "diffusion_regime" in res


def test_detect_latent_finbert_tone_shift():
    from src.sentiment_analysis import (
        detect_latent_finbert_tone_shift,
        compute_jensen_shannon_divergence,
    )
    import numpy as np

    p = np.array([0.7, 0.1, 0.2])
    q = np.array([0.1, 0.7, 0.2])
    js_div = compute_jensen_shannon_divergence(p, q)
    assert js_div > 0.20

    res = detect_latent_finbert_tone_shift(
        baseline_distribution=[0.60, 0.15, 0.25],
        current_distribution=[0.10, 0.70, 0.20],
    )
    assert res["is_narrative_pivot_detected"] is True
    assert "BEARISH" in res["tone_shift_verdict"]
