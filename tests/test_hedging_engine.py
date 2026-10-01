import pytest
from src.hedging_engine import (
    calculate_synthetic_floor_hedge,
    calculate_vix_call_ratio_backspread,
)


def test_calculate_synthetic_floor_hedge():
    res = calculate_synthetic_floor_hedge(
        portfolio_equity=150000.0,
        spot_index=500.0,
        floor_pct=0.95,
        volatility=0.18,
        time_to_expiry_years=0.25,
        barrier_pct=0.85,
    )

    assert res["status"] == "ACTIVE"
    assert res["hedge_active"] is True
    assert 0.0 <= res["recommended_hedge_ratio"] <= 1.0
    assert res["floor_level_dollars"] == 142500.0
    assert res["hedge_short_dollars"] > 0.0


def test_calculate_synthetic_floor_hedge_barrier_knockout():
    # Spot index at 400 with barrier at 425 (0.85 * 500) -> Knocked out
    res = calculate_synthetic_floor_hedge(
        portfolio_equity=150000.0,
        spot_index=400.0,
        floor_pct=0.95,
        barrier_pct=1.05,  # Artificially high barrier
    )

    assert res["status"] == "BARRIER_KNOCKED_OUT"
    assert res["hedge_active"] is False
    assert res["hedge_ratio"] == 0.0


def test_calculate_vix_call_ratio_backspread():
    # Sell 1x 18C @ $2.40, Buy 2x 24C @ $0.90 -> Net cost: 2*0.90 - 2.40 = -$0.60 (Credit of $60/contract)
    res = calculate_vix_call_ratio_backspread(
        vix_spot=16.0,
        strike_short=18.0,
        strike_long=24.0,
        ratio=2,
        premium_short=2.40,
        premium_long=0.90,
        contracts_short=5,
    )

    assert res["status"] == "SUCCESS"
    assert res["is_net_credit_entry"] is True
    assert res["net_entry_cash_flow_dollars"] > 0.0
    # In a crash where VIX reaches 40 and 75, payout must be strongly positive
    assert res["pnl_scenarios_dollars"]["VIX_40"] > 0.0
    assert (
        res["pnl_scenarios_dollars"]["VIX_75"] > res["pnl_scenarios_dollars"]["VIX_40"]
    )
    assert res["upper_breakeven_vix"] > 24.0
