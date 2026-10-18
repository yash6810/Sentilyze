"""
Unit Tests for Jev (TypeSafe RLCD Fast-Catalyst) Engine.
Validates:
1. Typed schema compliance (JevCatalystVerdict).
2. Sub-second execution latency (< 15ms per headline).
3. Bullish catalyst recognition (FDA approval, earnings beat, buyout).
4. Bearish probe recognition and hard CRO veto (DOJ probe, short seller attack).
5. Priority ranking (VETO supersedes neutral headlines).
"""

import pytest
from src.jev_catalyst import (
    JevCatalystEngine,
    JevCatalystVerdict,
    get_jev_catalyst_engine,
    evaluate_jev_catalyst_gate,
)


def test_jev_bullish_catalyst_classification():
    """Verify that FDA approval triggers Tier 1 Max Conviction."""
    engine = JevCatalystEngine()
    headline = "FDA clears novel oncology therapy following successful phase 3 endpoint"
    verdict = engine.classify_headline(ticker="MRK", headline=headline)

    assert isinstance(verdict, JevCatalystVerdict)
    assert verdict.material_catalyst is True
    assert verdict.event_type == "FDA_APPROVAL_EXPEDITED"
    assert verdict.calibrated_prob_up >= 0.85
    assert verdict.conviction_tier == "TIER_1_MAX_CONVICTION"
    assert verdict.immediate_action == "EXECUTE_IMMEDIATE_SCALE_IN"
    assert verdict.latency_ms < 15.0  # Ultra-low latency guarantee


def test_jev_bearish_veto_classification():
    """Verify that regulatory probe triggers Tier 4 Hard Veto."""
    engine = JevCatalystEngine()
    headline = "DOJ launches fraud investigation into accounting irregularities"
    verdict = engine.classify_headline(ticker="XYZ", headline=headline)

    assert verdict.material_catalyst is True
    assert verdict.event_type == "REGULATORY_PROBE_FRAUD"
    assert verdict.tail_risk_probability >= 0.80
    assert verdict.conviction_tier == "TIER_4_VETO"
    assert verdict.immediate_action == "HARD_VETO_EXIT"


def test_jev_neutral_noise_classification():
    """Verify that routine consumer noise is classified as neutral."""
    engine = JevCatalystEngine()
    headline = "Company announces new pastel colored silicone watch bands"
    verdict = engine.classify_headline(ticker="AAPL", headline=headline)

    assert verdict.material_catalyst is False
    assert verdict.event_type == "NEUTRAL_MARKET_NOISE"
    assert verdict.conviction_tier == "TIER_3_MONITOR"
    assert verdict.immediate_action == "IGNORE_NOISE"


def test_jev_priority_hierarchy_in_multi_headlines():
    """Verify that a probe in one headline vetoes the ticker even if other headlines are neutral."""
    engine = JevCatalystEngine()
    headlines = [
        "Company announces quarterly dividend payment date",
        "SEC opens inquiry into revenue recognition accounting",
        "Company CEO speaks at technology conference",
    ]
    verdict = engine.evaluate_ticker_catalysts(ticker="TEST", headlines=headlines)

    assert verdict.conviction_tier == "TIER_4_VETO"
    assert verdict.immediate_action == "HARD_VETO_EXIT"


def test_evaluate_jev_catalyst_gate_interface():
    """Verify the functional gatekeeper interface."""
    verdict = evaluate_jev_catalyst_gate(
        ticker="NVDA",
        headlines=[
            "NVIDIA raises full-year guidance after blowout AI data center revenue"
        ],
    )
    assert verdict.ticker == "NVDA"
    assert verdict.conviction_tier == "TIER_1_MAX_CONVICTION"
    assert verdict.calibrated_prob_up > 0.80
