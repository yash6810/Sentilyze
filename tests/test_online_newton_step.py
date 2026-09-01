import pytest
import numpy as np
import pandas as pd
from src.online_newton_step import OnlineNewtonStepOptimizer


def test_online_newton_step():
    ons = OnlineNewtonStepOptimizer(num_assets=3, eta=0.5, beta=1.0)
    assert len(ons.w) == 3
    assert ons.w.sum() == pytest.approx(1.0)

    # Step through 5 price relative updates
    for _ in range(5):
        price_relatives = np.array([1.02, 0.99, 1.01])
        new_w = ons.step(price_relatives)
        assert len(new_w) == 3
        assert new_w.sum() == pytest.approx(1.0)
        assert (new_w >= 0.0).all()


def test_ons_backtest_sequence():
    dates = pd.date_range("2025-01-01", periods=20)
    rets_df = pd.DataFrame(
        {
            "NVDA": [0.01] * 20,
            "AAPL": [-0.005] * 20,
            "MSFT": [0.002] * 20,
        },
        index=dates,
    )
    ons = OnlineNewtonStepOptimizer(num_assets=3)
    res = ons.backtest_sequence(rets_df)
    assert len(res) == 20
    assert "daily_return" in res.columns
    assert "total" in res.columns
    assert res["total"].iloc[-1] > 1.0
