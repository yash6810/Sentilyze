"""
Unit tests for High-Watermark Peak Profit Ratchet (75% Lock Floor).
"""

import pytest
from src.smart_trader_engine import apply_high_watermark_profit_lock


def test_high_watermark_locks_75pct_of_peak_gain():
    entry_price = 100.0
    initial_sl = 95.0

    # Stock rallies to 110.0 (+10% / +$1,000 profit on 100 shares)
    current_price = 110.0
    highest_seen = 110.0

    new_sl, new_peak, action = apply_high_watermark_profit_lock(
        current_price=current_price,
        entry_price=entry_price,
        highest_price_seen=highest_seen,
        current_sl=initial_sl,
        min_profit_threshold_pct=1.5,
        lock_fraction=0.75,
    )

    # Peak profit = +$10.00/share -> 75% locked = +$7.50/share
    # New Stop Floor MUST be at least $107.50
    assert new_sl >= 107.50
    assert new_peak == 110.0
    assert "75PCT_LOCK" in action


def test_high_watermark_does_not_lower_sl_on_pullback():
    entry_price = 100.0
    previous_sl = 107.50
    highest_seen = 110.0

    # Stock pulls back to 108.0 (from 110.0 peak)
    current_price = 108.0

    new_sl, new_peak, action = apply_high_watermark_profit_lock(
        current_price=current_price,
        entry_price=entry_price,
        highest_price_seen=highest_seen,
        current_sl=previous_sl,
        min_profit_threshold_pct=1.5,
        lock_fraction=0.75,
    )

    # Stop must remain locked at $107.50 (Never decreases!)
    assert new_sl == 107.50
    assert new_peak == 110.0
