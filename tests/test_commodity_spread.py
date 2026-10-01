"""
Unit tests for Commodity Convenience Yield & Roll Alpha (Sprint 2, Module 2.7)
"""

import pytest
from src.commodity_spread import (
    compute_convenience_yield,
    compute_roll_yield,
    analyze_commodity_term_structure,
    evaluate_macro_commodity_inflation_regime,
)


def test_compute_convenience_yield_backwardation():
    # Spot $80, 1-mo Futures $78 (Backwardation)
    y = compute_convenience_yield(
        spot_price=80.0,
        futures_price=78.0,
        tenor_years=1.0 / 12.0,
        risk_free_rate=0.045,
        storage_cost=0.03,
    )
    # y = 0.045 + 0.03 - 12 * ln(78/80) = 0.075 - 12 * (-0.0253) ~ +0.378 > 0.075
    assert y > 0.10


def test_compute_convenience_yield_contango():
    # Spot $80, 1-mo Futures $85 (Steep Contango)
    y = compute_convenience_yield(
        spot_price=80.0,
        futures_price=85.0,
        tenor_years=1.0 / 12.0,
        risk_free_rate=0.045,
        storage_cost=0.03,
    )
    assert y < 0.0


def test_compute_roll_yield():
    # Front $80, Next $76 -> Positive roll yield
    roll = compute_roll_yield(front_price=80.0, next_price=76.0, dt_days=30)
    assert roll > 0.0

    # Front $76, Next $80 -> Negative roll yield (drag)
    roll_neg = compute_roll_yield(front_price=76.0, next_price=80.0, dt_days=30)
    assert roll_neg < 0.0


def test_analyze_commodity_term_structure_smoke():
    res = analyze_commodity_term_structure("CRUDE_OIL")
    assert res["status"] == "SUCCESS"
    assert "convenience_yield_pct" in res
    assert "term_structure_regime" in res
    assert "macro_inflation_signal" in res


def test_evaluate_macro_commodity_inflation_regime():
    macro = evaluate_macro_commodity_inflation_regime()
    assert "overall_regime" in macro
    assert "details" in macro
    assert "CRUDE_OIL" in macro["details"]
