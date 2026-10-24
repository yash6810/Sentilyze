"""Unit Tests for Institutional Smart Order Routing & Adaptive Execution Router.

Verifies:
1. Amihud illiquidity and Kyle's Lambda metrics calculation.
2. Almgren-Chriss market impact and slippage estimation.
3. VWAP U-curve order slicing with exact share conservation.
4. TWAP jittered schedule generation.
5. ISOR decision logic (DIRECT_LIMIT vs VWAP vs TWAP).
6. Functional convenience API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adaptive_execution_router import (
    AdaptiveExecutionRouter,
    ChildOrderSlice,
    ExecutionPlan,
    route_institutional_order,
)


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Generate 30 bars of market history with realistic volume and prices."""
    np.random.seed(42)
    n = 30
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    volumes = np.random.randint(500000, 2000000, size=n)

    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices + 0.1,
            "Volume": volumes,
        }
    )


def test_microstructure_metrics(sample_market_df: pd.DataFrame):
    """Verify Amihud illiquidity and Kyle's lambda calculations."""
    router = AdaptiveExecutionRouter()

    amihud = router.estimate_amihud_illiquidity(sample_market_df)
    assert 0.0001 <= amihud <= 50.0

    kyle = router.estimate_kyle_lambda(sample_market_df, current_price=100.0)
    assert 1e-6 <= kyle <= 0.10


def test_almgren_chriss_market_impact_scaling():
    """Verify market impact scales monotonically with order size relative to ADV."""
    router = AdaptiveExecutionRouter()
    adv = 1_000_000.0
    price = 150.0

    impact_small = router.estimate_market_impact_bps(
        shares=100, price=price, adv_shares=adv
    )
    impact_med = router.estimate_market_impact_bps(
        shares=10_000, price=price, adv_shares=adv
    )
    impact_large = router.estimate_market_impact_bps(
        shares=100_000, price=price, adv_shares=adv
    )

    assert impact_small < impact_med < impact_large
    assert impact_small >= 0.5
    assert impact_large <= 250.0


def test_vwap_slices_share_conservation():
    """Verify VWAP slicing exactly preserves total share quantity without loss or truncation."""
    router = AdaptiveExecutionRouter()
    total_shares = 12_345
    current_price = 135.50

    slices = router.create_vwap_slices(
        total_shares=total_shares,
        current_price=current_price,
        action="BUY",
        num_slices=6,
        horizon_minutes=30,
    )

    assert len(slices) == 6
    sum_shares = sum(s.shares for s in slices)
    assert sum_shares == total_shares, f"Expected {total_shares}, got {sum_shares}"

    # Verify U-curve: first slice should be larger than middle slices
    assert slices[0].shares > slices[3].shares
    assert slices[0].limit_price > current_price  # Buy limit above market


def test_twap_slices_jitter_and_offsets():
    """Verify TWAP slicing with jitter produces valid monotonically ordered or bounded time offsets."""
    router = AdaptiveExecutionRouter()
    total_shares = 5000
    current_price = 220.0

    slices = router.create_twap_slices(
        total_shares=total_shares,
        current_price=current_price,
        action="SELL",
        num_slices=5,
        horizon_minutes=25,
        add_jitter=True,
    )

    assert len(slices) == 5
    sum_shares = sum(s.shares for s in slices)
    assert sum_shares == total_shares

    # Ensure all offsets are non-negative and bounded by horizon seconds
    max_offset = 25 * 60
    for s in slices:
        assert 0 <= s.offset_seconds <= max_offset
        assert s.limit_price < current_price  # Sell limit below market


def test_isor_routing_decisions(sample_market_df: pd.DataFrame):
    """Verify smart routing chooses DIRECT_LIMIT for tiny orders and VWAP for large orders."""
    router = AdaptiveExecutionRouter()

    # Tiny order (10 shares of high ADV stock) -> DIRECT_LIMIT
    plan_tiny = router.generate_execution_plan(
        ticker="AAPL",
        action="BUY",
        shares=10,
        current_price=230.0,
        history_df=sample_market_df,
    )
    assert plan_tiny.strategy == "DIRECT_LIMIT"
    assert plan_tiny.num_slices == 1

    # Large institutional order (50,000 shares) -> VWAP
    plan_large = router.generate_execution_plan(
        ticker="AAPL",
        action="BUY",
        shares=50_000,
        current_price=230.0,
        history_df=sample_market_df,
    )
    assert plan_large.strategy == "VWAP"
    assert plan_large.num_slices >= 4
    assert plan_large.estimated_impact_bps > plan_tiny.estimated_impact_bps


def test_functional_route_api(sample_market_df: pd.DataFrame):
    """Verify top-level route_institutional_order API."""
    data = route_institutional_order(
        ticker="NVDA",
        action="SELL",
        shares=1000,
        current_price=120.0,
        history_df=sample_market_df,
        strategy="TWAP",
    )

    assert isinstance(data, dict)
    assert data["ticker"] == "NVDA"
    assert data["strategy"] == "TWAP"
    assert data["total_shares"] == 1000
    assert len(data["child_slices"]) > 1
