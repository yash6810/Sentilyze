import pytest
import numpy as np
import pandas as pd
from src.triple_convex_engine import TripleConvexEngine


def test_triple_convex_engine_evaluate():
    dates = pd.date_range("2025-01-01", periods=60)
    data = {
        "NVDA": pd.DataFrame(
            {
                "Open": np.linspace(100, 130, 60),
                "High": np.linspace(102, 132, 60),
                "Low": np.linspace(99, 129, 60),
                "Close": np.linspace(101, 131, 60),
            },
            index=dates,
        ),
        "AAPL": pd.DataFrame(
            {
                "Open": np.linspace(150, 160, 60),
                "High": np.linspace(152, 162, 60),
                "Low": np.linspace(149, 159, 60),
                "Close": np.linspace(151, 161, 60),
            },
            index=dates,
        ),
    }

    engine = TripleConvexEngine(max_weight_per_asset=0.60)
    res = engine.evaluate_universe(data, vix_level=16.0)

    assert res["status"] == "OPTIMAL"
    assert "optimal_weights" in res
    assert res["optimal_weights"].sum() == pytest.approx(1.0, abs=1e-2)
    assert res["fractional_kelly_pct"] > 0.0
    assert res["solver_latency_ms"] < 500.0


def test_triple_convex_backtest():
    dates = pd.date_range("2025-01-01", periods=50)
    data = {
        "NVDA": pd.DataFrame(
            {
                "High": np.linspace(102, 120, 50),
                "Low": np.linspace(99, 118, 50),
                "Close": np.linspace(101, 119, 50),
            },
            index=dates,
        ),
        "MSFT": pd.DataFrame(
            {
                "High": np.linspace(202, 215, 50),
                "Low": np.linspace(199, 213, 50),
                "Close": np.linspace(201, 214, 50),
            },
            index=dates,
        ),
    }

    engine = TripleConvexEngine(max_weight_per_asset=0.60)
    bt_res = engine.backtest_multi_period(data, initial_capital=10000.0)

    assert len(bt_res) == 49
    assert "portfolio_value" in bt_res.columns
    assert "daily_return" in bt_res.columns
    assert bt_res["portfolio_value"].iloc[-1] > 10000.0
