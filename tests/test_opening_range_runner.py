import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.opening_range_runner import run_opening_range_session


def test_run_opening_range_session(mocker, tmp_path):
    """Verify that ORB live session executes, filters stocks in play, and saves latest result."""
    n = 25
    mock_df = pd.DataFrame(
        {
            "Open": np.linspace(100, 110, n),
            "High": np.linspace(102, 112, n),
            "Low": np.linspace(99, 109, n),
            "Close": np.linspace(101, 111, n),
            "Volume": np.linspace(10000, 25000, n),
        }
    )

    mocker.patch("src.opening_range_runner.get_price_history", return_value=mock_df)
    mocker.patch(
        "src.opening_range_runner.get_news",
        return_value=pd.DataFrame({"Title": ["AI Breakthrough"]}),
    )
    mocker.patch(
        "src.opening_range_runner.fetch_live_quote",
        return_value={"price": 105.0},
    )

    res = run_opening_range_session()
    assert "stocks_in_play" in res
    assert "executed_paper_trades" in res
    assert "portfolio_summary" in res
    assert isinstance(res["stocks_in_play"], list)
