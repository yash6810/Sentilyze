"""
Unit tests for OpenBB Institutional Data Bridge (src/openbb_bridge.py).
Verifies FRED macro regime classification, SEC Form 4 insider flow, and options fallback.
"""

import pytest
from src.openbb_bridge import (
    OpenBBBridge,
    MacroRegimeSnapshot,
    InsiderFlowSnapshot,
    OptionsSentimentSnapshot,
)


def test_openbb_bridge_macro_classification():
    bridge = OpenBBBridge()

    # Inverted yield curve
    regime, mult = bridge._classify_macro_regime(
        y10=3.8, y2=4.2, spread=-0.4, hy_spread=3.5
    )
    assert "YIELD_CURVE_INVERTED" in regime
    assert mult <= 0.75

    # Credit stress
    regime, mult = bridge._classify_macro_regime(
        y10=4.1, y2=3.8, spread=0.3, hy_spread=6.2
    )
    assert "CREDIT_STRESS" in regime
    assert mult <= 0.65

    # Healthy expansion
    regime, mult = bridge._classify_macro_regime(
        y10=4.2, y2=3.8, spread=0.4, hy_spread=3.0
    )
    assert "HEALTHY_EXPANSION" in regime
    assert mult >= 1.1


def test_openbb_bridge_live_or_fallback_macro():
    bridge = OpenBBBridge()
    macro = bridge.get_macro_regime()
    assert isinstance(macro, MacroRegimeSnapshot)
    assert macro.yield_10y > 0
    assert macro.yield_2y > 0
    assert macro.risk_on_multiplier > 0


def test_openbb_bridge_insider_flow():
    bridge = OpenBBBridge()
    insider = bridge.get_insider_transactions("NVDA")
    assert isinstance(insider, InsiderFlowSnapshot)
    assert insider.ticker == "NVDA"
    assert -1.0 <= insider.net_insider_score <= 1.0


def test_openbb_bridge_options_sentiment():
    bridge = OpenBBBridge()
    opts = bridge.get_options_sentiment("AAPL")
    assert isinstance(opts, OptionsSentimentSnapshot)
    assert opts.ticker == "AAPL"
    assert opts.put_call_volume_ratio >= 0
