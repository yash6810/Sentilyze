"""
Unit tests for Neural Additive Models (NAMs) (Sprint 3, Module 3.3)
"""

import pytest
import numpy as np
import torch
from src.models.nam_model import NeuralAdditiveModel, NAMClassifier


def test_nam_model_forward():
    model = NeuralAdditiveModel(input_dim=4, feature_names=["f1", "f2", "f3", "f4"])
    x = torch.randn(8, 4)
    logits, contribs = model(x)

    assert logits.shape == (8, 1)
    assert contribs.shape == (8, 4)
    # Total logit should equal bias + sum of contributions
    expected_logits = model.bias + torch.sum(contribs, dim=-1, keepdim=True)
    assert torch.allclose(logits, expected_logits)


def test_nam_classifier_fit_and_shape():
    np.random.seed(42)
    X = np.random.randn(80, 3)
    # Monotonic synthetic target: y increases with X[:, 0]
    y = (X[:, 0] > 0).astype(int)

    clf = NAMClassifier(
        input_dim=3,
        feature_names=["rsi", "momentum", "sentiment"],
        monotonic_indices=[0],  # Enforce monotonic constraint on feature 0
    )
    clf.fit(X, y, epochs=15, batch_size=16)

    probs = clf.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    assert np.allclose(np.sum(probs, axis=1), np.ones(5))

    shape_fn = clf.extract_shape_function(feature_idx=0, n_grid=25)
    assert shape_fn["feature_name"] == "rsi"
    assert len(shape_fn["x_domain"]) == 25
    assert len(shape_fn["f_response"]) == 25
