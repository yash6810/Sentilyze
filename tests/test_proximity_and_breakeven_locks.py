"""
Unit tests for Institutional Zero-Giveback Breakeven Lock, Proximity Take-Profit Harvester,
and High-Cash Anti-Paralysis Deployment Valve.
"""

import os
import json
import pytest
from unittest.mock import patch
from src.autonomous_trader import AutonomousTradingEngine
from src.agent_committee import ChiefRiskOfficerAgent
from src.paper_broker import PaperBroker


@pytest.fixture
def isolated_broker(tmp_path):
    """Creates a hermetic PaperBroker instance inside tmp_path to strictly protect production portfolios."""
    port_file = os.path.join(tmp_path, "paper_portfolio.json")
    trades_file = os.path.join(tmp_path, "executed_trades.csv")
    state = {
        "initial_capital": 100000.0,
        "cash": 100000.0,
        "total_equity": 100000.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "open_positions": {},
        "closed_trades": [],
    }
    with open(port_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    broker = PaperBroker(portfolio_file=port_file, trades_file=trades_file)
    return broker


def test_zero_giveback_breakeven_lock_threshold(isolated_broker):
    """
    Verify that when an open position experiences a gain of >= +1.20%,
    the engine ratchets the stop-loss to at least breakeven + commission/slippage cushion (entry * 1.002).
    """
    entry_price = 100.00
    initial_sl = 96.50
    isolated_broker.state["open_positions"]["NVDA"] = {
        "ticker": "NVDA",
        "entry_price": entry_price,
        "current_price": 101.25,
        "shares": 10,
        "sl_target": initial_sl,
        "tp1_target": 110.00,
        "tp2_target": 120.00,
        "high_water_mark": 101.25,
        "highest_price_seen": 101.25,
        "scaled_out": False,
    }
    isolated_broker._save()

    engine = AutonomousTradingEngine(broker=isolated_broker)
    quotes = {"NVDA": {"ticker": "NVDA", "price": 101.25, "status": "LIVE"}}

    with (
        patch("src.autonomous_trader.fetch_universe_live_quotes", return_value=quotes),
        patch("src.autonomous_trader.fetch_live_quote", return_value=quotes["NVDA"]),
        patch("src.autonomous_trader.send_discord_execution_alert"),
    ):
        actions = engine.manage_open_positions()

    pos = isolated_broker.state["open_positions"]["NVDA"]
    # Breakeven floor is at least entry * 1.002 = 100.20
    assert pos["sl_target"] >= 100.20


def test_proximity_tp1_harvester_50pct(isolated_broker):
    """
    Verify that if a trade reaches >= 98.5% of its TP1 target and begins pulling back,
    the 98.5% Proximity Harvester executes a 50% scale-out rather than risking giving back gains.
    """
    entry_price = 300.00
    tp1_target = 345.00  # Distance = $45.00. 10% pullback = $4.50
    peak_price = 344.20  # Reached 99.76% of TP1 (>= 98.5% = $339.825)
    pullback_price = 338.50  # Pulled back $5.70 from peak (>= $4.50 threshold)

    isolated_broker.state["open_positions"]["GD"] = {
        "ticker": "GD",
        "entry_price": entry_price,
        "current_price": pullback_price,
        "shares": 10,
        "sl_target": 292.00,
        "tp1_target": tp1_target,
        "tp2_target": 360.00,
        "high_water_mark": peak_price,
        "highest_price_seen": peak_price,
        "scaled_out": False,
    }
    isolated_broker._save()

    engine = AutonomousTradingEngine(broker=isolated_broker)
    quotes = {"GD": {"ticker": "GD", "price": pullback_price, "status": "LIVE"}}

    with (
        patch("src.autonomous_trader.fetch_universe_live_quotes", return_value=quotes),
        patch("src.autonomous_trader.fetch_live_quote", return_value=quotes["GD"]),
        patch("src.autonomous_trader.send_discord_execution_alert"),
    ):
        actions = engine.manage_open_positions()

    # Verify that a 50% scale-out occurred under TP1_PROXIMITY_HARVEST_50PCT
    assert len(actions["take_profits_tp1"]) >= 1
    tp1_exec = actions["take_profits_tp1"][0]
    assert tp1_exec["action"] == "TP1_PROXIMITY_HARVEST_50PCT"
    assert tp1_exec["shares_sold"] > 0
    assert tp1_exec["return_pct"] > 0

    # Verify that remaining position is marked scaled_out and stop moved to breakeven floor
    pos = isolated_broker.state["open_positions"]["GD"]
    assert pos["scaled_out"] is True
    assert pos["sl_target"] >= 300.00


def test_high_cash_deployment_valve_cro_approval():
    """
    Verify that when the portfolio has >= 80% liquid cash, the High-Cash Dynamic
    Deployment Valve activates to prevent multi-day capital stagnation, granting
    a 0.5x Kelly starter seed for candidates with 2 buy votes and >= 54.0% conviction.
    """
    cro = ChiefRiskOfficerAgent()

    # Create testimonies with 2 BUY votes and 54.3% consensus conviction (between 54.0% and 55.0%)
    mock_reports = [
        {
            "agent_name": "Technical Momentum Specialist",
            "vote": "BUY",
            "conviction_score": 60.0,
        },
        {
            "agent_name": "Sentiment & Alternative Data Specialist",
            "vote": "BUY",
            "conviction_score": 55.0,
        },
        {
            "agent_name": "Forensic & Valuation Auditor",
            "vote": "HOLD",
            "conviction_score": 48.0,
            "key_metrics": {"data_available": True},
        },
    ]

    with (
        patch(
            "src.finjepa_engine.evaluate_finjepa_physics_gate",
            return_value={"cro_veto": False, "regime": "NORMAL"},
        ),
        patch(
            "src.vpin_toxicity.evaluate_ticker_toxicity",
            return_value={
                "vpin": 0.30,
                "toxicity_regime": "LOW_TOXICITY_BENIGN",
                "is_toxic_dumping": False,
                "cro_veto_recommended": False,
            },
        ),
    ):
        signoff = cro.evaluate_and_sign_off("AAPL", 220.0, mock_reports, vix_level=16.0)

    # Valve should approve SCALE_IN with the anti-paralysis label
    assert signoff["action_code"] == "SCALE_IN"
    assert "HIGH-CASH STARTER SEED" in signoff["final_resolution"]
    assert signoff["kelly_allocation_pct"] > 0.0
