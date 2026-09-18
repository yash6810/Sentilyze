"""
Unit tests for Persistent Agent Memory Store (src/agent_memory.py).
Follows strict safety rules: uses isolated tmp_path directory.
"""

import os
import json
import pytest
from src.agent_memory import AgentMemoryStore, DEFAULT_AGENT_WEIGHTS


def test_agent_memory_initialization(tmp_path):
    mem_dir = str(tmp_path / "agent_memory")
    store = AgentMemoryStore(memory_dir=mem_dir)

    assert os.path.exists(store.rules_file)
    assert os.path.exists(store.weights_file)

    rules = store.get_semantic_rules()
    assert len(rules) >= 3

    weights = store.get_calibrated_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-4


def test_agent_memory_record_postmortem_and_recalibration(tmp_path):
    mem_dir = str(tmp_path / "agent_memory")
    store = AgentMemoryStore(memory_dir=mem_dir)

    # Record 5 simulated postmortems
    store.record_postmortem(
        ticker="NVDA",
        direction="LONG",
        agent_votes={
            "Technical Momentum": "BUY",
            "FinBERT Sentiment": "BUY",
            "Chief Risk Officer": "BUY",
        },
        outcome="WIN",
        r_multiple=2.1,
        pnl_pct=5.2,
        notes="Clean ORB breakout",
    )

    store.record_postmortem(
        ticker="AAPL",
        direction="LONG",
        agent_votes={
            "Technical Momentum": "BUY",
            "FinBERT Sentiment": "BUY",
            "Chief Risk Officer": "HOLD",
        },
        outcome="WIN",
        r_multiple=1.8,
        pnl_pct=3.1,
        notes="Earnings continuation",
    )

    store.record_postmortem(
        ticker="TSLA",
        direction="LONG",
        agent_votes={
            "Technical Momentum": "BUY",
            "FinBERT Sentiment": "BUY",
            "Adversarial Red-Team": "SELL",
        },
        outcome="LOSS",
        r_multiple=-1.0,
        pnl_pct=-2.5,
        notes="False breakout, Red-Team was right",
    )

    recent = store.get_recent_postmortems(limit=5)
    assert len(recent) == 3
    assert recent[0]["ticker"] == "TSLA"  # Reverse chronological

    # Verify weights sum to 1.0
    calibrated = store.get_calibrated_weights()
    assert abs(sum(calibrated.values()) - 1.0) < 1e-3
    assert "Technical Momentum" in calibrated


def test_agent_memory_add_semantic_rule(tmp_path):
    mem_dir = str(tmp_path / "agent_memory")
    store = AgentMemoryStore(memory_dir=mem_dir)

    rule = store.add_semantic_rule(
        description="Never buy ahead of Jackson Hole symposium",
        regime="MACRO_FED",
        confidence=0.92,
    )

    assert rule["rule_id"].startswith("RULE-")
    assert rule["regime"] == "MACRO_FED"

    rules = store.get_semantic_rules()
    assert any(r["rule_id"] == rule["rule_id"] for r in rules)
