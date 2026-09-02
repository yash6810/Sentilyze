import pytest
import numpy as np
import pandas as pd
from src.portfolio_diversity import calculate_portfolio_diversity_grade


def test_empty_portfolio():
    res = calculate_portfolio_diversity_grade([])
    assert res["status"] == "EMPTY_PORTFOLIO"
    assert res["grade"] == "N/A"
    assert res["average_correlation"] == 0.0


def test_single_asset_portfolio():
    res = calculate_portfolio_diversity_grade(["NVDA"])
    assert res["status"] == "SINGLE_ASSET_CONCENTRATION"
    assert res["grade"] == "D"
    assert res["effective_bets"] == 1.0


def test_custom_returns_diverse():
    # Construct 3 uncorrelated random return series
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100)
    df = pd.DataFrame(
        {
            "ASSET_A": np.random.normal(0, 0.02, 100),
            "ASSET_B": np.random.normal(0, 0.02, 100),
            "ASSET_C": np.random.normal(0, 0.02, 100),
        },
        index=dates,
    )
    res = calculate_portfolio_diversity_grade(
        ["ASSET_A", "ASSET_B", "ASSET_C"], custom_returns=df
    )
    assert res["status"] == "SUCCESS"
    assert res["grade"] in ["A+", "A-"]
    assert res["average_correlation"] < 0.20
    assert res["effective_bets"] >= 2.5


def test_custom_returns_correlated():
    # Construct 3 highly correlated series
    np.random.seed(42)
    base = np.random.normal(0, 0.02, 100)
    dates = pd.date_range("2026-01-01", periods=100)
    df = pd.DataFrame(
        {
            "TECH_1": base + np.random.normal(0, 0.001, 100),
            "TECH_2": base + np.random.normal(0, 0.001, 100),
            "TECH_3": base + np.random.normal(0, 0.001, 100),
        },
        index=dates,
    )
    res = calculate_portfolio_diversity_grade(
        ["TECH_1", "TECH_2", "TECH_3"], custom_returns=df
    )
    assert res["status"] == "SUCCESS"
    assert res["grade"] in ["C", "D"]
    assert res["average_correlation"] > 0.80
