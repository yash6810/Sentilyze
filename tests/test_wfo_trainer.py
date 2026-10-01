import pytest
import numpy as np
import pandas as pd
import torch
from src.wfo_trainer import (
    FinancialMLP,
    ElasticWeightConsolidation,
    train_model_ewc,
    run_continual_wfo_simulation,
)


def test_elastic_weight_consolidation_penalty():
    torch.manual_seed(42)
    np.random.seed(42)

    model = FinancialMLP(input_dim=5, hidden_dim=16, output_dim=1)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=50).astype(np.float32)

    # Initial training on Task 1
    train_model_ewc(model, X, y, epochs=10, lr=0.01)

    ewc = ElasticWeightConsolidation(model, X, y)

    # When model is unchanged, penalty should be 0.0
    pen_zero = ewc.penalty(model)
    assert pytest.approx(pen_zero.item(), abs=1e-5) == 0.0

    # Perturb weights
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)

    pen_perturbed = ewc.penalty(model)
    assert pen_perturbed.item() > 0.0


def test_run_continual_wfo_simulation():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.randint(0, 2, size=n),
        }
    )

    res = run_continual_wfo_simulation(
        df=df,
        feature_cols=["f1", "f2", "f3"],
        target_col="target",
        train_window=60,
        test_window=20,
        lambda_ewc=50.0,
        epochs=5,
    )

    assert res["n_steps"] >= 4
    assert res["total_oos_samples"] >= 80
    assert 0.0 <= res["accuracy"] <= 1.0
    assert len(res["ewc_predictions"]) == res["total_oos_samples"]
