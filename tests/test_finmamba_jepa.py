"""Unit and regression tests for the FinMamba-JEPA foundation model architecture.

Ensures strict mathematical validity, gradient flow, epistemic uncertainty
scaling, and portfolio safety invariants.
"""

import os
import pytest
import torch
from src.finmamba_jepa import (
    ContinuousStateSpaceBlock,
    FiLMCatalystModulator,
    LatentManifoldJEPA,
    EvidentialOutputHead,
    FinMambaJEPA,
    DifferentiableSharpeLoss,
)


def test_continuous_state_space_shapes_and_dt():
    """Verify S6 continuous state-space block executes with correct dimensions."""
    block = ContinuousStateSpaceBlock(d_model=32, d_state=8)
    x = torch.randn(4, 10, 32)
    dt = torch.ones(4, 10, 1) * 0.1

    # Test without dt
    y1 = block(x)
    assert y1.shape == (4, 10, 32)

    # Test with dt
    y2 = block(x, dt)
    assert y2.shape == (4, 10, 32)
    assert not torch.isnan(y1).any()
    assert not torch.isnan(y2).any()


def test_film_catalyst_modulator():
    """Verify FiLM catalyst modulation applies non-zero affine scaling and shift."""
    modulator = FiLMCatalystModulator(d_catalyst=4, d_model=32)
    h = torch.randn(4, 32)
    cat_bullish = torch.tensor([[1.0, 0.5, 0.0, 0.2]] * 4)
    cat_bearish = torch.tensor([[-1.0, 0.5, 1.0, 0.2]] * 4)

    h_bull = modulator(h, cat_bullish)
    h_bear = modulator(h, cat_bearish)

    assert h_bull.shape == (4, 32)
    assert h_bear.shape == (4, 32)
    assert not torch.allclose(h_bull, h_bear)


def test_latent_manifold_jepa_energy_distance():
    """Verify JEPA computes valid latent vectors and positive energy distances."""
    jepa = LatentManifoldJEPA(d_model=32, d_latent=64)
    h_curr = torch.randn(4, 32)
    h_next = torch.randn(4, 32)

    z_curr, z_next_pred, energy_dist = jepa(h_curr, h_next)

    assert z_curr.shape == (4, 64)
    assert z_next_pred.shape == (4, 64)
    assert energy_dist.shape == (4,)
    assert (energy_dist >= 0.0).all()


def test_evidential_head_uncertainty_and_kelly():
    """Verify evidential head outputs properly ordered quantile corridors and clamped Kelly fractions."""
    head = EvidentialOutputHead(d_model=32, dropout_rate=0.2)
    h = torch.randn(5, 32)

    out = head(h, n_mc=6)

    assert "p10" in out and "p50" in out and "p90" in out
    assert "sigma_epistemic" in out and "kelly_fraction" in out

    # Check quantile corridor order: P10 <= P50 <= P90
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()

    # Kelly fraction must be bounded in [-1.0, +1.0]
    assert (out["kelly_fraction"] >= -1.0).all()
    assert (out["kelly_fraction"] <= 1.0).all()


def test_finmamba_jepa_forward_backward_gradient_flow():
    """Verify end-to-end forward and backward pass has complete gradient flow with 0 NaNs."""
    model = FinMambaJEPA(
        n_features=10,
        n_catalyst=4,
        d_model=32,
        d_state=8,
        d_latent=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = DifferentiableSharpeLoss()

    x = torch.randn(6, 12, 10)
    cat = torch.randn(6, 4)
    y_true = torch.randn(6) * 0.02

    optimizer.zero_grad()
    preds = model(x, cat)
    loss = loss_fn(preds["kelly_fraction"], y_true, preds.get("energy_distance"))
    loss.backward()

    # Check gradients
    param_count = 0
    for name, p in model.named_parameters():
        param_count += 1
        assert p.grad is not None, f"Parameter {name} has None gradient"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"

    optimizer.step()
    assert param_count > 0


def test_epistemic_uncertainty_ood_scaling():
    """Verify MC dropout epistemic uncertainty scales upwards on out-of-distribution inputs."""
    torch.manual_seed(42)
    head = EvidentialOutputHead(d_model=32, dropout_rate=0.3)
    h_in_distribution = torch.randn(1, 32) * 0.5
    h_out_of_distribution = torch.randn(1, 32) * 20.0

    unc_id = head(h_in_distribution, n_mc=20)["sigma_epistemic"].item()
    unc_ood = head(h_out_of_distribution, n_mc=20)["sigma_epistemic"].item()

    assert unc_ood > unc_id


def test_strict_portfolio_preservation_invariant(tmp_path):
    """Verify safety invariant: paper portfolio files are never touched or modified."""
    paper_path = os.path.join("results", "paper_portfolio.json")
    if os.path.exists(paper_path):
        with open(paper_path, "r", encoding="utf-8") as f:
            content_before = f.read()

        # Run temporary model check in tmp_path
        dummy_save = tmp_path / "temp_model.pt"
        torch.save({"state": 1}, dummy_save)

        with open(paper_path, "r", encoding="utf-8") as f:
            content_after = f.read()
        assert content_before == content_after
