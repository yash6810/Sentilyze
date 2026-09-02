"""
Unit tests for Paper 25: Opening Range Breakout.
"""

import numpy as np
import pandas as pd
import pytest
from src.opening_range_breakout import OpeningRangeBreakout


def test_orb_filter_stocks_in_play():
    orb = OpeningRangeBreakout()
    np.random.seed(42)
    universe = {
        "NVDA": pd.DataFrame(
            {
                "Close": np.linspace(100, 150, 30),
                "Volume": np.random.uniform(1e6, 2e6, 30),
            }
        ),
        "AAPL": pd.DataFrame(
            {
                "Close": np.linspace(150, 155, 30),
                "Volume": np.random.uniform(5e5, 1e6, 30),
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "Close": np.linspace(300, 310, 30),
                "Volume": np.random.uniform(8e5, 1.2e6, 30),
            }
        ),
    }
    top = orb.filter_stocks_in_play(
        universe, catalyst_scores={"NVDA": 0.9, "AAPL": 0.4, "MSFT": 0.5}, top_k=2
    )
    assert len(top) == 2
    assert "NVDA" in top


def test_orb_evaluate_signals():
    orb = OpeningRangeBreakout()
    df = pd.DataFrame(
        {
            "Open": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
            ],
            "High": [
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                120,
            ],
            "Low": [
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
            ],
            "Close": [
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                119,
            ],
        }
    )
    res = orb.evaluate_orb_signals(df, sentiment_score=0.8)
    assert res["signal"] == 1
    assert res["strength"] > 0


def test_orb_backtest():
    orb = OpeningRangeBreakout()
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "NVDA": np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            "AAPL": np.cumprod(1 + np.random.normal(0.0005, 0.015, 100)),
        }
    )
    res = orb.backtest_orb_strategy(df)
    assert res["total_return_pct"] != 0.0
    assert "max_drawdown_pct" in res
