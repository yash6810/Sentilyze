import pytest
import numpy as np
import pandas as pd
from src.convex_optimizer import PolyTimeConvexOptimizer


def test_polytime_convex_optimizer():
    optimizer = PolyTimeConvexOptimizer(risk_aversion=1.0, max_weight_per_asset=0.4)
    alphas = pd.Series([0.15, 0.08, 0.05], index=["NVDA", "AAPL", "MSFT"])
    cov = pd.DataFrame(
        [
            [0.09, 0.02, 0.01],
            [0.02, 0.04, 0.01],
            [0.01, 0.01, 0.03],
        ],
        index=["NVDA", "AAPL", "MSFT"],
        columns=["NVDA", "AAPL", "MSFT"],
    )

    res = optimizer.optimize_allocation(alphas, cov)
    assert res["solver_success"] is True
    assert res["runtime_ms"] < 500.0  # Must run in sub-second polynomial time
    weights = res["weights"]
    assert len(weights) == 3
    assert weights.sum() == pytest.approx(1.0, abs=1e-2)
    assert (weights <= 0.40 + 1e-3).all()
    # Highest alpha asset (NVDA) should get the highest weight cap
    assert weights["NVDA"] >= weights["MSFT"]
