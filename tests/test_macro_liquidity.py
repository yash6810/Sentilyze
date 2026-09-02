import pytest
from src.macro_liquidity import calculate_macro_liquidity_metrics


def test_calculate_macro_liquidity_metrics():
    metrics = calculate_macro_liquidity_metrics()
    assert "10y_yield" in metrics
    assert "2y_yield" in metrics
    assert "spread_10_2_bps" in metrics
    assert "net_liquidity_trillions" in metrics
    assert metrics["net_liquidity_trillions"] > 0
    assert "financial_stress_score" in metrics
