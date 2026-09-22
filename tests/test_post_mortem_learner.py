"""
Unit tests for Autonomous Trade Post-Mortem & Bayesian Council Recalibration Engine.
Strictly isolated using pytest's tmp_path to protect production portfolio state.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.post_mortem_learner import (
    TradePostMortemLearner,
    DEFAULT_WEIGHTS,
)


@pytest.fixture
def mock_trades_env(tmp_path):
    """Creates isolated mock trades and weights in a temporary directory."""
    trades_file = tmp_path / "mock_executed_trades.csv"
    weights_file = tmp_path / "mock_committee_weights.json"
    memory_file = tmp_path / "mock_agent_learning_memory.json"
    portfolio_file = tmp_path / "mock_portfolio.json"

    # Create synthetic mock trades (3 winners, 1 controlled stop, 1 severe stop)
    mock_data = pd.DataFrame(
        [
            {
                "ticker": "GEV",
                "shares": 50,
                "entry_price": 880.0,
                "exit_price": 940.0,
                "entry_date": "2026-09-16",
                "exit_date": "2026-09-19",
                "pnl": 3000.0,
                "return_pct": 6.82,
                "exit_reason": "Target 1 Take-Profit Harvest",
            },
            {
                "ticker": "PTC",
                "shares": 100,
                "entry_price": 130.0,
                "exit_price": 136.5,
                "entry_date": "2026-09-14",
                "exit_date": "2026-09-15",
                "pnl": 650.0,
                "return_pct": 5.0,
                "exit_reason": "Target 2 Take-Profit Harvest",
            },
            {
                "ticker": "Q",
                "shares": 200,
                "entry_price": 110.0,
                "exit_price": 113.0,
                "entry_date": "2026-09-15",
                "exit_date": "2026-09-16",
                "pnl": 600.0,
                "return_pct": 2.73,
                "exit_reason": "MODEL_SELL",
            },
            {
                "ticker": "PGR",
                "shares": 100,
                "entry_price": 210.0,
                "exit_price": 207.9,
                "entry_date": "2026-09-20",
                "exit_date": "2026-09-21",
                "pnl": -210.0,
                "return_pct": -1.0,
                "exit_reason": "Protective Stop-Loss",
            },
            {
                "ticker": "BAD",
                "shares": 50,
                "entry_price": 100.0,
                "exit_price": 94.0,
                "entry_date": "2026-09-20",
                "exit_date": "2026-09-21",
                "pnl": -300.0,
                "return_pct": -6.0,
                "exit_reason": "STOP_LOSS",
            },
        ]
    )
    mock_data.to_csv(trades_file, index=False)

    return {
        "trades_csv": str(trades_file),
        "weights_json": str(weights_file),
        "memory_json": str(memory_file),
        "portfolio_json": str(portfolio_file),
    }


def test_learner_loads_trades(mock_trades_env):
    learner = TradePostMortemLearner(
        trades_csv_path=mock_trades_env["trades_csv"],
        weights_path=mock_trades_env["weights_json"],
        memory_path=mock_trades_env["memory_json"],
    )
    df = learner.load_executed_trades()
    assert len(df) == 5
    assert "GEV" in df["ticker"].values


def test_trade_categorization(mock_trades_env):
    learner = TradePostMortemLearner(
        trades_csv_path=mock_trades_env["trades_csv"],
        weights_path=mock_trades_env["weights_json"],
        memory_path=mock_trades_env["memory_json"],
    )
    df = learner.load_executed_trades()

    winner = learner.categorize_trade(df.iloc[0])
    assert winner["quality"] == "WINNER"
    assert winner["classification"] == "ALPHA_HARVEST"

    shield = learner.categorize_trade(df.iloc[3])
    assert shield["quality"] == "CONTROLLED_LOSS"
    assert shield["classification"] == "CAPITAL_SHIELD_DEFENSE"

    severe = learner.categorize_trade(df.iloc[4])
    assert severe["quality"] == "SEVERE_LOSS"


def test_bayesian_weight_update_invariance(mock_trades_env):
    learner = TradePostMortemLearner(
        trades_csv_path=mock_trades_env["trades_csv"],
        weights_path=mock_trades_env["weights_json"],
        memory_path=mock_trades_env["memory_json"],
        learning_rate=0.2,
    )

    prior = DEFAULT_WEIGHTS.copy()
    deltas = {
        "Technical Momentum": 0.5,
        "FinBERT Sentiment": 0.5,
        "Fundamental Valuation": 0.1,
        "Chief Risk Officer": 0.3,
        "Adversarial Red-Team": -0.2,
    }

    posterior = learner.update_weights_bayesian(prior, deltas)

    # 1. Weights must sum to exactly 1.0
    assert abs(sum(posterior.values()) - 1.0) < 1e-4

    # 2. Weights must respect floors (0.10) and ceilings (0.35)
    for k, v in posterior.items():
        assert 0.099 <= v <= 0.351

    # 3. High-performing specialists should gain weight
    assert posterior["Technical Momentum"] >= prior["Technical Momentum"]
    assert posterior["FinBERT Sentiment"] >= prior["FinBERT Sentiment"]


def test_full_post_mortem_cycle(mock_trades_env):
    learner = TradePostMortemLearner(
        trades_csv_path=mock_trades_env["trades_csv"],
        weights_path=mock_trades_env["weights_json"],
        memory_path=mock_trades_env["memory_json"],
    )

    report = learner.run_post_mortem_cycle()

    assert report["total_trades_analyzed"] == 5
    assert report["win_rate_pct"] == 60.0
    assert report["winners_count"] == 3
    assert report["controlled_losses_count"] == 1
    assert report["severe_losses_count"] == 1

    # Verify atomic files were created
    assert os.path.exists(mock_trades_env["weights_json"])
    assert os.path.exists(mock_trades_env["memory_json"])

    with open(mock_trades_env["weights_json"], "r") as f:
        saved_weights = json.load(f)
        assert abs(sum(saved_weights.values()) - 1.0) < 1e-4
