"""
Unit tests for Max Compound Acceleration Engine.
"""

import pytest
from src.compound_engine import (
    calculate_doubling_progress,
    compute_compound_position_size,
)


def test_calculate_doubling_progress():
    res = calculate_doubling_progress(initial_capital=100000.0, current_equity=125000.0)
    assert res["progress_pct"] == 25.0
    assert res["net_gain_dollars"] == 25000.0
    assert res["goal_dollars_remaining"] == 75000.0
    assert len(res["milestones"]) == 5
    assert res["milestones"][1]["status"] == "COMPLETED 🟢"


def test_compute_compound_position_size():
    sizing = compute_compound_position_size(
        current_total_equity=150000.0,
        available_cash=100000.0,
        confidence=0.80,
        max_position_fraction=0.25,
    )
    assert sizing["allocated_dollars"] > 0
    assert sizing["allocated_dollars"] <= 150000.0 * 0.25
    assert sizing["max_risk_dollars"] == 150000.0 * 0.025
