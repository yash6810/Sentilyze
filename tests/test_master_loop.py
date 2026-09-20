"""
Unit Tests for Sentilyze Master Autonomous Trading Loop.

Verifies:
1. Phase 1 Pre-Market Reconnaissance with mock macro data.
2. Phase 2 Market Open Whipsaw Shield.
3. Phase 3 Intraday Execution & Arbitration with Circuit Breakers.
4. Phase 4 Post-Market Autopsy & Calibration.
5. Strict portfolio preservation (isolated in tmp_path).
"""

import os
import json
import pytest
from unittest.mock import MagicMock
from src.master_loop import MasterTradingLoop
from src.paper_broker import PaperBroker


@pytest.fixture
def mock_isolated_broker(tmp_path):
    """Creates an isolated PaperBroker pointing to a temporary file in tmp_path."""
    test_port_file = str(tmp_path / "test_paper_portfolio.json")
    broker = PaperBroker(portfolio_path=test_port_file, initial_cash=100000.0)
    return broker


def test_master_loop_initialization(mock_isolated_broker):
    """Tests that MasterTradingLoop initializes with isolated broker and components."""
    loop = MasterTradingLoop(broker=mock_isolated_broker)
    assert loop.broker is mock_isolated_broker
    assert loop.alpaca is not None
    assert loop.institutional_gateway is not None
    assert loop.sec_crawler is not None


def test_phase_1_premarket(mocker, mock_isolated_broker):
    """Tests Phase 1 Pre-Market Reconnaissance."""
    loop = MasterTradingLoop(broker=mock_isolated_broker)

    mocker.patch.object(
        loop.institutional_gateway,
        "get_macro_regime",
        return_value={
            "regime": "EXPANSION",
            "vix": 14.2,
            "yield_spread": 0.25,
        },
    )
    mocker.patch(
        "src.master_loop.evaluate_macro_volatility_blackout",
        return_value={"is_blackout_active": False, "reason": "No scheduled FOMC/CPI"},
    )
    mocker.patch.object(
        loop.sec_crawler,
        "fetch_recent_8k_filings",
        return_value=[{"filing_type": "8-K", "material_risk_flag": False}],
    )
    mocker.patch(
        "src.master_loop.compile_biotech_catalyst_radar",
        return_value={"risk_profile": "LOW"},
    )
    mocker.patch.object(
        loop.engine,
        "run_premarket_briefing",
        return_value={"status": "success", "discord_dispatched": True},
    )

    res = loop.run_premarket_phase(watchlist=["NVDA", "AAPL"])
    assert res["phase"] == "PRE_MARKET_RECONNAISSANCE"
    assert res["macro_regime"]["regime"] == "EXPANSION"
    assert res["volatility_blackout"]["is_blackout_active"] is False
    assert "NVDA" in res["catalysts_scanned"]


def test_phase_2_opening_shield(mock_isolated_broker):
    """Tests Phase 2 Opening Range Whipsaw Shield."""
    loop = MasterTradingLoop(broker=mock_isolated_broker)

    # Test with bypass flag on
    res = loop.run_opening_shield_phase(bypass_time_check=True)
    assert res["phase"] == "MARKET_OPEN_WHIPSAW_SHIELD"
    assert res["shield_active"] is True
    assert res["new_entries_allowed"] is False

    # Test with bypass flag off (simulating standard market conditions)
    res_off = loop.run_opening_shield_phase(bypass_time_check=False)
    assert "holdings_guarded_count" in res_off


def test_phase_3_intraday_kill_switch_active(mocker, mock_isolated_broker):
    """Tests that Phase 3 halts when the Master Kill Switch is active."""
    loop = MasterTradingLoop(broker=mock_isolated_broker)
    mocker.patch("src.master_loop.is_kill_switch_active", return_value=True)

    res = loop.run_intraday_phase(candidate_tickers=["NVDA"])
    assert res["status"] == "HALTED_BY_KILL_SWITCH"
    assert res["trades_executed"] == 0


def test_phase_3_intraday_daily_loss_circuit_breaker(mocker, mock_isolated_broker):
    """Tests that Phase 3 halts when the max daily loss circuit breaker is triggered."""
    loop = MasterTradingLoop(broker=mock_isolated_broker)
    mocker.patch("src.master_loop.is_kill_switch_active", return_value=False)
    mocker.patch("src.master_loop.check_daily_loss_circuit_breaker", return_value=True)

    res = loop.run_intraday_phase(candidate_tickers=["NVDA"])
    assert res["status"] == "HALTED_BY_DAILY_LOSS_BREAKER"
    assert res["trades_executed"] == 0


def test_phase_3_intraday_execution_flow(mocker, mock_isolated_broker):
    """Tests normal committee deliberation and execution flow in Phase 3."""
    loop = MasterTradingLoop(broker=mock_isolated_broker)
    mocker.patch("src.master_loop.is_kill_switch_active", return_value=False)
    mocker.patch("src.master_loop.check_daily_loss_circuit_breaker", return_value=False)

    mock_delib = {
        "final_resolution": "BUY",
        "consensus_conviction_pct": 72.0,
        "cro_veto_triggered": False,
        "target_ticker": "NVDA",
    }
    mocker.patch("src.master_loop.convene_trading_committee", return_value=mock_delib)
    mocker.patch(
        "src.master_loop.execute_committee_order",
        return_value={
            "status": "EXECUTED",
            "ticker": "NVDA",
            "shares": 10,
            "entry_price": 120.0,
            "tp1": 130.0,
            "sl": 115.0,
        },
    )
    mocker.patch.object(loop.alpaca, "is_connected", return_value=True)
    mock_alpaca_submit = mocker.patch.object(
        loop.alpaca,
        "submit_bracket_order",
        return_value={"status": "FILLED", "order_id": "alpaca_123"},
    )

    res = loop.run_intraday_phase(candidate_tickers=["NVDA"])
    assert res["status"] == "COMPLETED"
    assert len(res["executed_orders"]) == 1
    mock_alpaca_submit.assert_called_once()


def test_phase_4_postmarket(mocker, mock_isolated_broker, tmp_path):
    """Tests Phase 4 Post-Market Autopsy and Calibration."""
    test_trades_file = str(tmp_path / "test_executed_trades.csv")
    loop = MasterTradingLoop(broker=mock_isolated_broker, trades_path=test_trades_file)

    mocker.patch(
        "src.master_loop.run_autonomous_trade_autopsy",
        return_value={"status": "SUCCESS", "win_rate": 85.0, "total_trades": 10},
    )
    mocker.patch.object(
        loop.engine,
        "_run_self_improvement_feedback_loop",
        return_value={"total_cycles": 1, "agent_weights": {"Technical": 0.25}},
    )

    res = loop.run_postmarket_phase(force_autopsy=True)
    assert res["phase"] == "POST_MARKET_AUTOPSY"
    assert res["status"] == "COMPLETED"
    assert res["trade_autopsy"]["win_rate"] == 85.0


def test_full_daily_cycle_isolated(mocker, mock_isolated_broker, tmp_path):
    """Tests run_full_daily_cycle with complete phase chaining in isolation."""
    test_trades_file = str(tmp_path / "test_trades.csv")
    loop = MasterTradingLoop(broker=mock_isolated_broker, trades_path=test_trades_file)

    # Patch phases
    mocker.patch.object(
        loop,
        "run_premarket_phase",
        return_value={"phase": "PRE_MARKET_RECONNAISSANCE"},
    )
    mocker.patch.object(
        loop,
        "run_opening_shield_phase",
        return_value={"phase": "MARKET_OPEN_WHIPSAW_SHIELD"},
    )
    mocker.patch.object(
        loop, "run_intraday_phase", return_value={"phase": "INTRADAY_EXECUTION"}
    )
    mocker.patch.object(
        loop,
        "run_postmarket_phase",
        return_value={"phase": "POST_MARKET_AUTOPSY", "final_equity": 100000.0},
    )

    cycle_res = loop.run_full_daily_cycle(dry_run=True, ignore_market_hours=True)
    assert cycle_res["portfolio_preserved"] is True
    assert cycle_res["phase_1_premarket"]["phase"] == "PRE_MARKET_RECONNAISSANCE"
    assert cycle_res["phase_2_opening_shield"]["phase"] == "MARKET_OPEN_WHIPSAW_SHIELD"
    assert cycle_res["phase_3_intraday"]["phase"] == "INTRADAY_EXECUTION"
    assert cycle_res["phase_4_postmarket"]["phase"] == "POST_MARKET_AUTOPSY"


def test_strict_live_portfolio_never_touched():
    """Verifies that the live paper_portfolio.json exists and is strictly intact."""
    live_file = os.path.join("results", "paper_portfolio.json")
    assert os.path.exists(live_file)
    with open(live_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_equity"] >= 150000.0
    assert data["cash"] >= 150000.0
