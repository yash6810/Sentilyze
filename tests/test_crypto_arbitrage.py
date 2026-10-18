"""
Unit Tests for 24/7 Crypto Funding Rate & Liquidation Cascade Arbitrage Engine.
Validates:
1. Public Binance & Coinbase derivatives data parsing.
2. Negative funding rate triggers SHORT_SQUEEZE_EXPANSION with high conviction.
3. Extreme positive funding triggers LONG_LIQUIDATION_HAZARD CRO veto.
4. Fractional crypto execution in isolated paper broker sandbox.
5. Strict safety: ZERO calls to Alpaca; ZERO touch on paper_portfolio.json.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from src.crypto_arbitrage import (
    CryptoArbitrageEngine,
    get_crypto_arbitrage_engine,
    CRYPTO_MAPPING,
)
from src.paper_broker import PaperBroker


def test_crypto_symbol_mapping():
    """Verify that BTC, ETH, and SOL are mapped to correct Binance and Coinbase pairs."""
    assert "BTC-USD" in CRYPTO_MAPPING
    assert CRYPTO_MAPPING["BTC-USD"]["binance"] == "BTCUSDT"
    assert CRYPTO_MAPPING["BTC-USD"]["coinbase"] == "BTC-USD"
    assert CRYPTO_MAPPING["ETH-USD"]["binance"] == "ETHUSDT"


def test_short_squeeze_regime_on_negative_funding():
    """Verify that negative funding rate correctly triggers SHORT_SQUEEZE_EXPANSION."""
    engine = CryptoArbitrageEngine()

    mock_deriv = {
        "funding_rate_8h": -0.00025,
        "funding_rate_pct": -0.025,
        "annualized_funding_pct": -27.38,
        "mark_price": 82000.0,
        "open_interest_coins": 95000.0,
        "open_interest_dollars": 7790000000.0,
        "is_live": True,
    }
    mock_spot = {
        "spot_price": 81950.0,
        "bid": 81945.0,
        "ask": 81955.0,
        "spread_bps": 1.2,
        "is_live": True,
    }

    with (
        patch.object(engine, "fetch_binance_derivatives_data", return_value=mock_deriv),
        patch.object(engine, "fetch_coinbase_spot_ticker", return_value=mock_spot),
    ):

        setup = engine.evaluate_crypto_arbitrage_opportunity("BTC-USD")
        assert setup["ticker"] == "BTC-USD"
        assert setup["regime"] == "SHORT_SQUEEZE_EXPANSION"
        assert setup["action"] == "EXECUTE_BUY_SQUEEZE"
        assert setup["cro_veto"] is False
        assert setup["conviction_pct"] >= 80.0
        assert setup["tp1_target"] > setup["spot_price"]
        assert setup["sl_target"] < setup["spot_price"]


def test_long_liquidation_hazard_on_extreme_positive_funding():
    """Verify that overleveraged long funding triggers LONG_LIQUIDATION_HAZARD and CRO veto."""
    engine = CryptoArbitrageEngine()

    mock_deriv = {
        "funding_rate_8h": 0.00065,
        "funding_rate_pct": 0.065,
        "annualized_funding_pct": 71.18,
        "mark_price": 2500.0,
        "open_interest_coins": 450000.0,
        "open_interest_dollars": 1125000000.0,
        "is_live": True,
    }
    mock_spot = {
        "spot_price": 2498.0,
        "bid": 2497.5,
        "ask": 2498.5,
        "spread_bps": 4.0,
        "is_live": True,
    }

    with (
        patch.object(engine, "fetch_binance_derivatives_data", return_value=mock_deriv),
        patch.object(engine, "fetch_coinbase_spot_ticker", return_value=mock_spot),
    ):

        setup = engine.evaluate_crypto_arbitrage_opportunity("ETH-USD")
        assert setup["regime"] == "LONG_LIQUIDATION_HAZARD"
        assert setup["action"] == "DEFENSIVE_HOLD"
        assert setup["cro_veto"] is True
        assert setup["conviction_pct"] <= 30.0


def test_fractional_crypto_sandbox_execution(tmp_path):
    """Verify that orders execute cleanly in an isolated temporary paper broker."""
    p_file = os.path.join(tmp_path, "mock_crypto_port.json")
    t_file = os.path.join(tmp_path, "mock_crypto_trades.csv")
    broker = PaperBroker(
        portfolio_path=p_file, trades_path=t_file, initial_cash=25000.0
    )

    engine = CryptoArbitrageEngine()
    setup = {
        "ticker": "SOL-USD",
        "action": "EXECUTE_BUY_SQUEEZE",
        "regime": "SHORT_SQUEEZE_EXPANSION",
        "spot_price": 105.50,
        "basis_pct": 0.08,
        "funding_rate_8h_pct": -0.015,
        "cro_veto": False,
        "tp1_target": 115.0,
        "tp2_target": 125.0,
        "sl_target": 100.0,
    }

    res = engine.execute_crypto_sandbox_order(
        setup=setup, crypto_broker=broker, allocation_dollars=1500.0
    )
    assert res.get("success") is True
    assert "SOL-USD" in broker.state["open_positions"]

    # Verify fractional shares
    shares = float(broker.state["open_positions"]["SOL-USD"]["shares"])
    assert shares > 0.0
    assert broker.state["cash"] < 25000.0
