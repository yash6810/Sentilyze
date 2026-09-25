"""
Unit Tests for AgentMemoryStore & Episodic Trade Memory.
========================================================
Validates:
1. Initializing episodic memory directories and baseline rules.
2. Recording trade post-mortems and calculating R-multiples.
3. Dynamic agent committee weight recalibration.
4. Ticker reputation tracking (consecutive losses, win rate, total PnL).
5. Memory risk adjustments (penalties, caution levels, super-majority flags).
6. Syncing from executed trades CSV.
7. Automated post-mortem recording via PaperBroker._record_trade_closure.
"""

import os
import pytest
import pandas as pd
from src.agent_memory import AgentMemoryStore
from src.paper_broker import PaperBroker


@pytest.fixture
def mock_memory(tmp_path):
    mem_dir = str(tmp_path / "agent_memory")
    return AgentMemoryStore(memory_dir=mem_dir)


def test_agent_memory_initialization(mock_memory, tmp_path):
    """Verifies that baseline rules and default weights are created upon initialization."""
    assert os.path.exists(mock_memory.rules_file)
    assert os.path.exists(mock_memory.weights_file)

    rules = mock_memory.get_semantic_rules()
    assert len(rules) >= 3

    weights = mock_memory.get_calibrated_weights()
    assert "Technical Momentum" in weights
    assert "FinBERT Sentiment" in weights


def test_record_postmortem_and_recalibration(mock_memory):
    """Verifies appending post-mortems and updating empirical weights."""
    mock_memory.record_postmortem(
        ticker="NVDA",
        direction="LONG",
        agent_votes={"Technical Momentum": "BUY", "FinBERT Sentiment": "BUY"},
        outcome="WIN",
        r_multiple=2.5,
        pnl_pct=5.2,
        pnl_dollars=520.0,
        exit_reason="TAKE_PROFIT",
        notes="High-conviction trend follower win.",
    )

    mock_memory.record_postmortem(
        ticker="PGR",
        direction="LONG",
        agent_votes={"Technical Momentum": "BUY", "FinBERT Sentiment": "HOLD"},
        outcome="LOSS",
        r_multiple=-1.0,
        pnl_pct=-2.5,
        pnl_dollars=-350.0,
        exit_reason="STOP_LOSS",
        notes="Protective stop triggered.",
    )

    recent = mock_memory.get_recent_postmortems(limit=10)
    assert len(recent) == 2
    assert recent[0]["ticker"] == "PGR"
    assert recent[1]["ticker"] == "NVDA"


def test_ticker_reputation_and_risk_adjustment(mock_memory):
    """Verifies reputation scoring and risk penalty calculations."""
    # Record two consecutive losses for a ticker
    mock_memory.record_postmortem(
        ticker="BADTICKER",
        outcome="LOSS",
        pnl_pct=-3.0,
        pnl_dollars=-600.0,
        exit_reason="STOP_LOSS",
    )
    mock_memory.record_postmortem(
        ticker="BADTICKER",
        outcome="LOSS",
        pnl_pct=-2.8,
        pnl_dollars=-550.0,
        exit_reason="STOP_LOSS",
    )

    rep = mock_memory.get_ticker_reputation("BADTICKER")
    assert rep["total_trades"] == 2
    assert rep["losses"] == 2
    assert rep["wins"] == 0
    assert rep["consecutive_losses"] == 2
    assert rep["total_pnl_dollars"] == -1150.0

    adjust = mock_memory.get_memory_risk_adjustment("BADTICKER")
    assert adjust["caution_level"] == "PENALIZED"
    assert adjust["require_supermajority"] is True
    assert adjust["conviction_delta"] < 0
    assert adjust["kelly_scale"] < 1.0


def test_alpha_champion_reputation(mock_memory):
    """Verifies that high-expectancy winners receive alpha champion boosts."""
    mock_memory.record_postmortem(
        ticker="PLTR",
        outcome="WIN",
        pnl_pct=50.0,
        pnl_dollars=3500.0,
        exit_reason="TP2_RUNNER_EXIT",
    )

    rep = mock_memory.get_ticker_reputation("PLTR")
    assert rep["total_trades"] == 1
    assert rep["wins"] == 1
    assert rep["win_rate"] == 1.0
    assert rep["total_pnl_dollars"] == 3500.0

    adjust = mock_memory.get_memory_risk_adjustment("PLTR")
    assert adjust["caution_level"] == "ALPHA_CHAMPION"
    assert adjust["conviction_delta"] > 0
    assert adjust["kelly_scale"] > 1.0
    assert adjust["require_supermajority"] is False


def test_sync_from_executed_trades(tmp_path):
    """Verifies synchronizing executed trades from an external CSV file."""
    mem_dir = str(tmp_path / "sync_mem")
    store = AgentMemoryStore(memory_dir=mem_dir)

    # Create dummy executed_trades.csv
    csv_file = str(tmp_path / "dummy_trades.csv")
    dummy_data = [
        {
            "ticker": "AAPL",
            "shares": 10,
            "entry_price": 150.0,
            "exit_price": 160.0,
            "entry_date": "2026-09-01",
            "exit_date": "2026-09-05",
            "pnl": 100.0,
            "return_pct": 6.67,
            "reason": "TAKE_PROFIT",
        },
        {
            "ticker": "TSLA",
            "shares": 5,
            "entry_price": 200.0,
            "exit_price": 190.0,
            "entry_date": "2026-09-02",
            "exit_date": "2026-09-04",
            "pnl": -50.0,
            "return_pct": -5.0,
            "reason": "STOP_LOSS",
        },
    ]
    pd.DataFrame(dummy_data).to_csv(csv_file, index=False)

    added = store.sync_from_executed_trades(csv_path=csv_file)
    assert added == 2

    # Second sync should add 0 duplicates
    added_again = store.sync_from_executed_trades(csv_path=csv_file)
    assert added_again == 0

    recent = store.get_recent_postmortems()
    assert len(recent) == 2


def test_broker_auto_records_postmortem(tmp_path):
    """Verifies that PaperBroker._record_trade_closure automatically records post-mortems."""
    p_file = str(tmp_path / "portfolio.json")
    broker = PaperBroker(portfolio_path=p_file, initial_cash=100000.0)

    trade_record = {
        "ticker": "GOOGL",
        "shares": 20,
        "entry_price": 140.0,
        "exit_price": 145.0,
        "entry_date": "2026-09-10",
        "exit_date": "2026-09-12",
        "pnl": 100.0,
        "return_pct": 3.57,
        "reason": "TAKE_PROFIT",
    }

    broker._record_trade_closure(trade_record)
    assert len(broker.state["closed_trades"]) == 1

    # Check that post-mortem was recorded in tmp_path/agent_memory
    mem_dir = str(tmp_path / "agent_memory")
    mem_store = AgentMemoryStore(memory_dir=mem_dir)
    recent = mem_store.get_recent_postmortems()
    assert len(recent) == 1
    assert recent[0]["ticker"] == "GOOGL"
    assert recent[0]["outcome"] == "WIN"
    assert recent[0]["pnl_dollars"] == 100.0
