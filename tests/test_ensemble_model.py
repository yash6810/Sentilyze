"""
Unit tests for 3-Way Super-Ensemble (XGBoost + LightGBM + CatBoost).
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.ensemble_model import SuperEnsembleClassifier


def test_super_ensemble_fit_predict():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f_{i}" for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, size=100))

    ensemble = SuperEnsembleClassifier()
    ensemble.fit(X, y)

    # Test Prediction & Probability
    proba = ensemble.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    preds = ensemble.predict(X)
    assert len(preds) == 100
    assert set(np.unique(preds)).issubset({0, 1})

    # Test Model Consensus
    consensus = ensemble.evaluate_model_consensus(X.tail(1))
    assert "consensus_agreement" in consensus
    assert consensus["total_models"] == 3
    assert "xgboost" in consensus["individual_models"]
    assert "lightgbm" in consensus["individual_models"]
    assert "catboost" in consensus["individual_models"]


def test_super_ensemble_save_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = os.path.join(tmp_dir, "test_ensemble_model.json")
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f"f_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, size=50))

        ensemble = SuperEnsembleClassifier()
        ensemble.fit(X, y)
        ensemble.save(base_path)

        # Load back
        loaded = SuperEnsembleClassifier()
        loaded.load(base_path)
        preds = loaded.predict(X)
        assert len(preds) == 50
