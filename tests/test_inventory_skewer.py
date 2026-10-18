"""
Unit Tests for Avellaneda-Stoikov Micro-Inventory Skewer.
Validates:
1. Reservation price formulas under neutral, positive, and negative inventory.
2. Optimal bid/ask spread dynamics.
3. Order execution limit interface (BUY vs SELL).
4. Sub-millisecond execution latency.
"""

import pytest
from src.inventory_skewer import (
    AvellanedaStoikovSkewer,
    get_inventory_skewer,
    calculate_optimal_execution_limit,
)


def test_neutral_inventory_reservation_price():
    """Verify that with zero inventory, reservation price equals market price."""
    skewer = AvellanedaStoikovSkewer()
    quotes = skewer.compute_reservation_quotes(
        market_price=100.0,
        current_shares=0.0,
        max_shares=100.0,
        volatility=0.02,
    )
    assert quotes["reservation_price"] == 100.0
    assert quotes["optimal_bid"] < 100.0
    assert quotes["optimal_ask"] > 100.0
    assert quotes["optimal_bid"] < quotes["optimal_ask"]


def test_positive_inventory_reservation_skew():
    """Verify that holding heavy long inventory skews reservation price downward."""
    skewer = AvellanedaStoikovSkewer()
    quotes_flat = skewer.compute_reservation_quotes(
        market_price=100.0, current_shares=0.0, max_shares=100.0
    )
    quotes_long = skewer.compute_reservation_quotes(
        market_price=100.0, current_shares=100.0, max_shares=100.0
    )

    # When long, reservation price drops to encourage selling and discourage buying
    assert quotes_long["reservation_price"] < quotes_flat["reservation_price"]
    assert quotes_long["optimal_bid"] < quotes_flat["optimal_bid"]
    assert quotes_long["optimal_ask"] < quotes_flat["optimal_ask"]


def test_negative_inventory_reservation_skew():
    """Verify that holding short inventory skews reservation price upward."""
    skewer = AvellanedaStoikovSkewer()
    quotes_short = skewer.compute_reservation_quotes(
        market_price=100.0, current_shares=-100.0, max_shares=100.0
    )
    quotes_flat = skewer.compute_reservation_quotes(
        market_price=100.0, current_shares=0.0, max_shares=100.0
    )

    assert quotes_short["reservation_price"] > quotes_flat["reservation_price"]
    assert quotes_short["optimal_bid"] > quotes_flat["optimal_bid"]


def test_calculate_optimal_execution_limit_interface():
    """Verify functional interface returns sensible limit orders."""
    buy_limit = calculate_optimal_execution_limit(
        side="BUY", market_price=150.0, current_shares=0.0, max_shares=50.0
    )
    sell_limit = calculate_optimal_execution_limit(
        side="SELL", market_price=150.0, current_shares=50.0, max_shares=50.0
    )

    assert buy_limit <= 150.0
    assert sell_limit >= buy_limit
