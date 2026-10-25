"""
Tests for Risk Controls and Position Limits in Sentilyze.
Verifies:
1. (a) A second buy for the same ticker is blocked (both local and Alpaca live).
2. (b) A position exceeding 10% of total portfolio equity is blocked.
3. (c) The 11th concurrent position is blocked (MAX_CONCURRENT_POSITIONS = 10).
"""

import pytest
from unittest.mock import patch, MagicMock
from src.paper_broker import PaperBroker


@pytest.fixture
def isolated_broker(tmp_path):
    portfolio_file = str(tmp_path / "paper_portfolio.json")
    trades_file = str(tmp_path / "executed_trades.csv")
    broker = PaperBroker(
        portfolio_path=portfolio_file,
        initial_cash=100000.0,
        trades_path=trades_file,
    )
    return broker


def test_second_buy_same_ticker_is_blocked(isolated_broker):
    """(a) A second buy for the same ticker is strictly blocked."""
    # Mock Alpaca as disconnected initially
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        # 1. Initial buy succeeds
        res1 = isolated_broker.execute_buy(ticker="AAPL", shares=10, price=150.0)
        assert res1["success"] is True
        assert "AAPL" in isolated_broker.state["open_positions"]

        # 2. Second buy for the same ticker in local ledger is blocked
        res2 = isolated_broker.execute_buy(ticker="AAPL", shares=5, price=155.0)
        assert res2["success"] is False
        assert "already exists" in res2["error"]

    # 3. Buy for ticker that has open position or open order on Alpaca is blocked
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[{"symbol": "MSFT", "market_value": "5000"}],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[],
        ),
    ):
        res3 = isolated_broker.execute_buy(ticker="MSFT", shares=10, price=200.0)
        assert res3["success"] is False
        assert "Alpaca" in res3["error"]

    # 4. Buy for ticker that has a pending open order on Alpaca is blocked
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[{"symbol": "GOOGL", "status": "new", "qty": "10"}],
        ),
    ):
        res4 = isolated_broker.execute_buy(ticker="GOOGL", shares=10, price=180.0)
        assert res4["success"] is False
        assert "Alpaca" in res4["error"]


def test_position_over_10_pct_equity_is_blocked(isolated_broker):
    """(b) A position exceeding 10% of total equity is blocked."""
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        # Total equity is $100,000. 10% cap is $10,000.
        # Attempting to buy 100 shares @ $150.0 ($15,000 = 15% of equity) must be blocked
        res_oversized = isolated_broker.execute_buy(
            ticker="NVDA", shares=100, price=150.0
        )
        assert res_oversized["success"] is False
        assert "exceeds max 10% of equity" in res_oversized["error"]

        # Buying 50 shares @ $150.0 ($7,500 = 7.5% of equity) must succeed
        res_valid = isolated_broker.execute_buy(ticker="NVDA", shares=50, price=150.0)
        assert res_valid["success"] is True
        assert "NVDA" in isolated_broker.state["open_positions"]

    # Also test when existing Alpaca pending order causes combined exposure > 10%
    with (
        patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=True),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_positions",
            return_value=[],
        ),
        patch(
            "src.alpaca_broker.AlpacaBrokerBridge.get_open_orders",
            return_value=[
                {"symbol": "AMZN", "status": "new", "qty": "50", "limit_price": 160.0}
            ],
        ),
    ):
        # Existing pending order is $8,000 (8%). Attempting to add $3,000 more (total $11,000 = 11%) is blocked
        res_combined_oversized = isolated_broker.execute_buy(
            ticker="AMZN", shares=20, price=150.0
        )
        assert res_combined_oversized["success"] is False


def test_eleventh_concurrent_position_is_blocked(isolated_broker):
    """(c) The 11th concurrent position is blocked when MAX_CONCURRENT_POSITIONS is 10."""
    with patch("src.alpaca_broker.AlpacaBrokerBridge.is_connected", return_value=False):
        # Fill exactly 10 distinct concurrent positions
        tickers = [f"TICK{i}" for i in range(1, 11)]
        for t in tickers:
            res = isolated_broker.execute_buy(ticker=t, shares=2, price=100.0)
            assert res["success"] is True, f"Failed to open position for {t}: {res}"

        assert len(isolated_broker.state["open_positions"]) == 10

        # Attempt to open an 11th concurrent position
        res_11th = isolated_broker.execute_buy(ticker="TICK11", shares=2, price=100.0)
        assert res_11th["success"] is False
        assert "Maximum concurrent positions (10) reached" in res_11th["error"]
        assert "TICK11" not in isolated_broker.state["open_positions"]
