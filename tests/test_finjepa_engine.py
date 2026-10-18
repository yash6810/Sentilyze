"""
Unit Tests for FinJEPA (Financial Joint-Embedding Predictive Architecture).
Validates:
1. PyTorch latent space representation and prediction forward pass.
2. Market feature extraction from price action DataFrames.
3. Latent displacement and regime physics classification.
4. Native PyTorch weights_only=True serialization security.
5. Functional gatekeeper evaluate_finjepa_physics_gate().
"""

import os
import numpy as np
import pandas as pd
import pytest
import torch

from src.finjepa_engine import (
    FinJEPAEngine,
    get_finjepa_engine,
    evaluate_finjepa_physics_gate,
)


def test_finjepa_network_forward_pass():
    """Verify encoder and predictor output dimensions."""
    engine = FinJEPAEngine(in_dim=10, latent_dim=16, cond_dim=4)
    x = np.random.randn(10).astype(np.float32)
    cond = np.random.randn(4).astype(np.float32)

    physics = engine.predict_latent_physics(x, cond)
    assert "latent_displacement" in physics
    assert "latent_cosine_alignment" in physics
    assert "regime" in physics
    assert "risk_verdict" in physics
    assert "physics_confidence" in physics
    assert isinstance(physics["cro_veto"], bool)
    assert -1.0 <= physics["latent_cosine_alignment"] <= 1.0


def test_finjepa_feature_extraction():
    """Verify that extract_market_features returns correct (10, 4) shapes."""
    engine = FinJEPAEngine()
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    prices = 100.0 + np.cumsum(np.random.randn(30) * 2.0)
    df = pd.DataFrame(
        {
            "Close": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Volume": np.random.randint(100000, 500000, 30),
        },
        index=dates,
    )

    x_state, u_cond = engine.extract_market_features(df)
    assert x_state.shape == (10,)
    assert u_cond.shape == (4,)
    assert not np.isnan(x_state).any()
    assert not np.isnan(u_cond).any()


def test_finjepa_evaluate_asset_interface():
    """Verify end-to-end evaluation on simulated price DataFrame."""
    np.random.seed(123)
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    prices = 150.0 + np.cumsum(np.random.randn(40) * 1.5)
    df = pd.DataFrame(
        {
            "Close": prices,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Volume": np.random.randint(500000, 2000000, 40),
        },
        index=dates,
    )

    result = evaluate_finjepa_physics_gate(ticker="TSLA", df=df)
    assert result["ticker"] == "TSLA"
    assert "regime" in result
    assert "latent_displacement" in result
    assert "physics_confidence" in result
    assert 0.0 <= result["physics_confidence"] <= 1.0


def test_finjepa_safe_serialization(tmp_path):
    """Verify safe PyTorch model serialization with weights_only=True."""
    weights_file = os.path.join(tmp_path, "test_finjepa.pt")
    engine = FinJEPAEngine(weights_path=weights_file)
    engine.save_weights(weights_file)
    assert os.path.exists(weights_file)

    # Re-instantiate with saved weights
    engine2 = FinJEPAEngine(weights_path=weights_file)
    assert engine2.encoder is not None
