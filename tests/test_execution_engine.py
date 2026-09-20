import numpy as np
import pytest
from src.execution_engine import (
    compute_almgren_chriss_trajectory,
    compute_vwap_schedule,
    compute_twap_schedule,
    compute_bouchaud_market_impact,
    route_smart_order,
)


def test_almgren_chriss_trajectory():
    total_shares = 10_000.0
    res = compute_almgren_chriss_trajectory(
        total_shares=total_shares,
        total_time_hours=6.5,
        n_intervals=13,
    )
    assert res["model"] == "Almgren-Chriss Optimal Execution"
    assert res["total_shares"] == total_shares
    assert len(res["schedule"]) == 13

    # Check sum of executed shares matches total order within rounding
    sum_shares = sum(item["shares_to_execute"] for item in res["schedule"])
    assert np.isclose(sum_shares, total_shares, atol=2.0)

    # Check monotonic decline of remaining holdings
    remainings = [item["remaining_shares"] for item in res["schedule"]]
    for i in range(len(remainings) - 1):
        assert remainings[i] >= remainings[i + 1]


def test_vwap_schedule_u_curve():
    total_shares = 5_000.0
    res = compute_vwap_schedule(total_shares=total_shares, n_intervals=13)
    schedule = res["schedule"]
    assert len(schedule) == 13

    # Check conservation of shares
    sum_shares = sum(item["shares_to_execute"] for item in schedule)
    assert np.isclose(sum_shares, total_shares, atol=1.0)

    # Check canonical U-shape: open (interval 1) and close (interval 13) have higher weight than midday (interval 7)
    open_pct = schedule[0]["pct_of_order"]
    midday_pct = schedule[6]["pct_of_order"]
    close_pct = schedule[-1]["pct_of_order"]

    assert open_pct > midday_pct
    assert close_pct > midday_pct


def test_twap_stealth_schedule():
    total_shares = 6_500.0
    res = compute_twap_schedule(
        total_shares=total_shares, n_intervals=13, randomize_pct=0.15
    )
    schedule = res["schedule"]
    assert len(schedule) == 13

    sum_shares = sum(item["shares_to_execute"] for item in schedule)
    assert np.isclose(sum_shares, total_shares, atol=1.0)

    # Check slices are not all strictly identical due to stealth anti-gaming randomization
    slices = [item["shares_to_execute"] for item in schedule]
    assert len(set(slices)) > 1


def test_bouchaud_market_impact():
    shares = 10_000.0
    spot = 200.0
    adv = 1_000_000.0

    impact = compute_bouchaud_market_impact(
        shares=shares,
        spot_price=spot,
        avg_daily_volume=adv,
        daily_volatility=0.02,
    )
    assert impact["effective_buy_price"] > spot
    assert impact["effective_sell_price"] < spot
    assert impact["impact_bps"] > 0.0
    assert impact["dollar_slippage"] > 0.0

    # Impact should increase with higher trade size (square-root scaling)
    impact_large = compute_bouchaud_market_impact(
        shares=40_000.0,
        spot_price=spot,
        avg_daily_volume=adv,
        daily_volatility=0.02,
    )
    assert impact_large["impact_bps"] > impact["impact_bps"]


def test_route_smart_order():
    sor = route_smart_order(
        ticker="NVDA",
        shares=2_500.0,
        spot_price=180.0,
        strategy="VWAP",
    )
    assert sor["ticker"] == "NVDA"
    assert sor["order_shares"] == 2_500.0
    assert "market_impact" in sor
    assert "execution_plan" in sor
