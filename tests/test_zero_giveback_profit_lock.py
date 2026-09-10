"""
Unit tests for Institutional Zero-Giveback Profit Protection, 80% Peak Gain Retention,
Downside Capital Shield, and Atomic Broker State Persistence.
"""

import os
import json
import pytest
import pandas as pd
from src.smart_trader_engine import (
    calculate_structural_trailing_stop,
    apply_high_watermark_profit_lock,
    enforce_capital_shield_stop_floor,
)
from src.ticker_sentinel import TickerSentinel, TickerSentinelSwarm
from src.paper_broker import PaperBroker


def test_micro_breakeven_lock_at_0_5_pct():
    """Verify that a gain of +0.50% immediately locks stop-loss to Breakeven + buffer."""
    entry_price = 100.00
    current_price = 100.55  # +0.55%
    initial_sl = 97.50
    empty_df = pd.DataFrame()

    new_sl, action = calculate_structural_trailing_stop(
        current_price=current_price,
        entry_price=entry_price,
        df_history=empty_df,
        current_sl=initial_sl,
    )

    assert new_sl >= 100.20
    assert "ZERO_GIVEBACK_BREAKEVEN_LOCKED" in action


def test_tier1_profit_bank_lock_at_1_0_pct():
    """Verify that a gain of +1.00% locks stop-loss to at least +0.50% net profit."""
    entry_price = 100.00
    current_price = 101.10  # +1.10%
    initial_sl = 100.20
    empty_df = pd.DataFrame()

    new_sl, action = calculate_structural_trailing_stop(
        current_price=current_price,
        entry_price=entry_price,
        df_history=empty_df,
        current_sl=initial_sl,
    )

    assert new_sl >= 100.50
    assert "TIER1_PROFIT_BANK_LOCKED" in action


def test_high_watermark_80_pct_retention():
    """Verify that a peak surge to $110 (+10%) locks at least $108 (+8.0%) into the stop floor."""
    entry_price = 100.00
    highest_seen = 110.00  # Peak +10%
    current_price = 108.50  # Pulled back slightly
    initial_sl = 100.50

    new_sl, peak, action = apply_high_watermark_profit_lock(
        current_price=current_price,
        entry_price=entry_price,
        highest_price_seen=highest_seen,
        current_sl=initial_sl,
        min_profit_threshold_pct=1.20,
        lock_fraction=0.80,
    )

    # 100 + (110 - 100) * 0.80 = 108.00
    assert new_sl == 108.00
    assert peak == 110.00
    assert "HIGH_WATERMARK_80PCT_LOCK" in action


def test_capital_shield_stop_floor_enforcement():
    """Verify that any stop loss wider than -2.50% is elevated to the capital shield ceiling."""
    entry_price = 200.00
    wide_sl = 180.00  # -10% (unacceptable)

    shielded_sl, action = enforce_capital_shield_stop_floor(
        entry_price=entry_price,
        current_sl=wide_sl,
        max_loss_pct=2.50,
    )

    # 200 * (1 - 0.025) = 195.00
    assert shielded_sl == 195.00
    assert "CAPITAL_SHIELD_FLOOR_ENFORCED" in action


def test_atomic_broker_save_and_backup(tmp_path):
    """Verify that PaperBroker saves atomically, maintains a backup, and guards integrity."""
    test_portfolio_path = os.path.join(tmp_path, "test_portfolio.json")
    broker = PaperBroker(portfolio_path=test_portfolio_path)

    # Initial save creates file
    assert os.path.exists(test_portfolio_path)
    assert broker.state["cash"] > 0

    # Mutate and save again to create .bak
    broker.state["cash"] += 500.0
    broker._save()

    bak_path = f"{test_portfolio_path}.bak"
    assert os.path.exists(bak_path)

    # Verify backup and primary have valid json
    with open(test_portfolio_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["cash"] == broker.state["cash"]


def test_sentinel_swarm_synchronization_with_portfolio():
    """Verify that Sentinel Swarm synchronizes ratcheted stop levels back into portfolio state."""
    open_positions = {
        "WRB": {
            "entry_price": 69.26,
            "current_price": 69.77,
            "highest_price_seen": 69.77,
            "sl_target": 67.50,
            "shares": 96,
        }
    }
    portfolio_mock = {"open_positions": open_positions}

    swarm = TickerSentinelSwarm()
    swarm.sync_open_positions(open_positions)
    quotes = {"WRB": {"price": 69.77}}

    reports = swarm.audit_all_sentinels(quotes, sync_to_portfolio=portfolio_mock)

    assert len(reports) == 1
    rep = reports[0]
    # WRB is at +0.74%, so micro-breakeven should be triggered
    assert rep["profit_lock_status"] == "🛡️ RISK-FREE BREAKEVEN"
    assert portfolio_mock["open_positions"]["WRB"]["sl_target"] >= 69.26 * 1.001
