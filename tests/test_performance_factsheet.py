import pytest
import numpy as np
import pandas as pd
from src.performance_factsheet import generate_comprehensive_factsheet


def test_generate_comprehensive_factsheet_default():
    factsheet = generate_comprehensive_factsheet()
    assert "total_return_pct" in factsheet
    assert "sharpe_ratio" in factsheet
    assert "sortino_ratio" in factsheet
    assert "calmar_ratio" in factsheet
    assert "max_drawdown_pct" in factsheet
    assert "monthly_grid_df" in factsheet
    assert "curves_df" in factsheet
    assert factsheet["total_trading_days"] > 100


def test_generate_comprehensive_factsheet_custom_series():
    dates = pd.date_range("2025-01-01", periods=150)
    ret = pd.Series(np.random.normal(0.001, 0.01, 150), index=dates)
    factsheet = generate_comprehensive_factsheet(returns_series=ret)
    assert factsheet["sortino_ratio"] is not None
    assert factsheet["calmar_ratio"] is not None
    assert factsheet["monthly_grid_df"].shape[0] >= 1
