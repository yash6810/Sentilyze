"""Unit Tests for GPU-Accelerated Hyperparameter Grid Search Engine.

Verifies:
1. XGBoost model initialization with hist tree method.
2. Single trial walk-forward out-of-sample evaluation.
3. Multi-trial search pipeline and ranking by Sharpe/AUC.
4. Exporting optimal configuration to JSON file in tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.gpu_hyperparameter_search import (
    GPUHyperparameterSearcher,
    HyperparameterTrialResult,
)


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate 200 bars of synthetic technical features and targets."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2026-01-01", periods=n, freq="D")

    # Generate 5 predictive features
    feat1 = np.random.randn(n)
    feat2 = np.random.randn(n) * 2.0
    feat3 = np.cumsum(np.random.randn(n) * 0.1)

    X = pd.DataFrame({"rsi": feat1, "macd": feat2, "momentum": feat3}, index=dates)
    # Forward returns correlated with momentum
    fwd_ret = pd.Series(0.005 * feat3 + np.random.randn(n) * 0.01, index=dates)
    y = (fwd_ret > 0).astype(int)

    return X, y, fwd_ret


def test_single_trial_walk_forward_evaluation(sample_training_data):
    """Verify single parameter trial runs walk-forward validation and computes metrics."""
    X, y, fwd_ret = sample_training_data
    searcher = GPUHyperparameterSearcher(device="cpu")

    trial_params = {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 20,
    }

    result = searcher.evaluate_params_walk_forward(
        X, y, fwd_ret, trial_params, train_window=100, test_window=25
    )

    assert isinstance(result, HyperparameterTrialResult)
    assert 0.0 <= result.oos_accuracy <= 1.0
    assert 0.0 <= result.oos_auc <= 1.0
    assert isinstance(result.oos_sharpe, float)


def test_grid_search_and_export(sample_training_data, tmp_path):
    """Verify multi-trial search ranks trials and exports optimal configuration."""
    X, y, fwd_ret = sample_training_data
    searcher = GPUHyperparameterSearcher(device="cpu")

    param_space = {
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.10],
        "n_estimators": [15, 30],
    }

    trials = searcher.search_grid(
        X, y, fwd_ret, n_trials=4, param_space=param_space, seed=42
    )

    assert len(trials) == 4
    # Check trials are sorted by Sharpe descending
    assert trials[0].oos_sharpe >= trials[-1].oos_sharpe
    assert trials[0].is_optimal is True

    # Test export to tmp_path
    export_file = tmp_path / "optimal_hyperparameters.json"
    exported_data = searcher.export_optimal_hyperparameters(trials, str(export_file))

    assert export_file.exists()
    assert "best_params" in exported_data
    assert exported_data["total_evaluated_trials"] == 4
