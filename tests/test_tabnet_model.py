"""
Unit tests for TabNet Attentive Interpretable Tabular Network (Sprint 3, Module 3.1)
"""

import pytest
import numpy as np
import torch
from src.models.tabnet_model import sparsemax, TabNetModel, TabNetClassifier


def test_sparsemax_properties():
    # Sparsemax should sum to 1.0 and produce exact zeros
    z = torch.tensor([[10.0, 0.0, -5.0], [1.0, 1.0, 1.0]])
    p = sparsemax(z, dim=-1)
    assert torch.allclose(torch.sum(p, dim=-1), torch.ones(2))
    # For [10, 0, -5], the smallest elements should be exact 0.0
    assert p[0, 1].item() == 0.0
    assert p[0, 2].item() == 0.0
    assert p[0, 0].item() == 1.0


def test_tabnet_model_forward():
    model = TabNetModel(input_dim=8, output_dim=2, n_steps=3)
    x = torch.randn(4, 8)
    logits, m_explain, step_masks = model(x)

    assert logits.shape == (4, 2)
    assert m_explain.shape == (4, 8)
    assert len(step_masks) == 3
    # Attention masks should sum to 1 across features
    assert torch.allclose(torch.sum(m_explain, dim=-1), torch.ones(4), atol=1e-4)


def test_tabnet_classifier_fit_explain():
    np.random.seed(42)
    X = np.random.randn(60, 6)
    # Synthetic target: feature 0 and 1 are informative
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    feat_names = [f"f_{i}" for i in range(6)]
    clf = TabNetClassifier(input_dim=6, n_steps=2, feature_names=feat_names)
    clf.fit(X, y, epochs=10, batch_size=16)

    probs = clf.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    explanation = clf.explain(X[:10])
    assert "global_feature_masks" in explanation
    assert len(explanation["global_feature_masks"]) == 6
    assert "top_driving_feature" in explanation
