"""
Unit tests for src/academic_papers_benchmark.py.
Verifies the master academic research empirical benchmark suite execution.
"""

import pytest
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.academic_papers_benchmark import (
    run_all_14_papers_benchmark,
    BENCHMARK_RESULTS_FILE,
)


@pytest.fixture
def mock_stock_data():
    dates = pd.date_range("2023-01-01", periods=120, freq="B")
    np.random.seed(42)
    # Generate smooth upward trending series with noise
    ret = np.random.normal(0.001, 0.015, size=len(dates))
    price = 100.0 * np.cumprod(1.0 + ret)
    df = pd.DataFrame(
        {
            "Open": price * 0.99,
            "High": price * 1.02,
            "Low": price * 0.98,
            "Close": price,
            "Volume": np.random.randint(1000000, 5000000, size=len(dates)),
        },
        index=dates,
    )
    return df


def test_benchmark_results_file_structure():
    """Verify that existing benchmark results file has required schema."""
    if os.path.exists(BENCHMARK_RESULTS_FILE):
        with open(BENCHMARK_RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "evaluation_period" in data
        assert "results" in data
        assert len(data["results"]) >= 10
        # Check first result structure
        first_key = list(data["results"].keys())[0]
        first_item = data["results"][first_key]
        assert "name" in first_item
        assert "total_return_pct" in first_item
        assert "annualized_sharpe" in first_item


def test_run_all_14_papers_benchmark_mocked(mock_stock_data, tmp_path):
    """Test run_all_14_papers_benchmark with mocked price feed."""
    with patch(
        "src.academic_papers_benchmark.get_price_history",
        return_value=mock_stock_data,
    ):
        test_out_file = str(tmp_path / "mock_benchmark_results.json")
        with patch(
            "src.academic_papers_benchmark.BENCHMARK_RESULTS_FILE",
            test_out_file,
        ):
            res = run_all_14_papers_benchmark(
                tickers=["AAPL", "MSFT"],
                lookback_period="6mo",
            )
            assert "results" in res
            assert isinstance(res["results"], dict)
            assert len(res["results"]) > 0
            assert os.path.exists(test_out_file)
