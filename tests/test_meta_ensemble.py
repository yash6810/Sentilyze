import numpy as np
import pandas as pd
from src.meta_ensemble import train_meta_ensemble


def _generate_synthetic_training_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series((X["feat_0"] + X["feat_1"] > 0).astype(int))
    return X, y


def test_meta_ensemble_fit_and_predict():
    X, y = _generate_synthetic_training_data()
    ensemble = train_meta_ensemble(X, y)

    assert ensemble.is_fitted
    preds = ensemble.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_meta_ensemble_voter_breakdown():
    X, y = _generate_synthetic_training_data()
    ensemble = train_meta_ensemble(X, y)

    proba, voter_breakdown = ensemble.predict_proba(X.tail(1))
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0

    assert "XGBoost (50% wt)" in voter_breakdown
    assert "Random Forest (30% wt)" in voter_breakdown
    assert "Logistic Regression (20% wt)" in voter_breakdown
    assert "Meta-Ensemble Consensus" in voter_breakdown
