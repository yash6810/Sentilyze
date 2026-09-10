"""
Unit tests for the Tri-Model Neural Brain (Fast FinBERT + Event Companion + DLinear-TCN + MoE Gating).
"""

import pytest
import numpy as np
import torch
from src.event_classifier_model import EventClassifierModel
from src.fast_finbert import FastFinBERTEngine, get_fast_finbert_engine
from src.neural_gating_network import NeuralGatingNetwork
from src.fast_agent_pipeline import FastAgentPipeline, get_fast_agent_pipeline


def test_event_classifier_model():
    model = EventClassifierModel()

    # Test M&A Detection
    m_and_a = model.classify(
        "NVIDIA to acquire chip company in $10B all-cash buyout deal"
    )
    assert m_and_a["event_type"] == "M_AND_A_TAKEOVER"
    assert m_and_a["urgency_score"] >= 0.70
    assert m_and_a["is_material"] is True
    assert m_and_a["latency_ms"] < 10.0

    # Test Slang Normalization
    normalized = model.normalize_social_text("🚀🚀🚀 diamond hands to the moon 💎🙌")
    assert "surge" in normalized or "rally" in normalized or "long" in normalized


def test_event_classifier_fraud_and_dilution():
    model = EventClassifierModel()

    # Test Fraud / DOJ Probe
    probe = model.classify(
        "SEC launches accounting fraud investigation and subpoenas executives"
    )
    assert probe["event_type"] == "LEGAL_FRAUD_INQUIRY"
    assert probe["sentiment_bias"] < -0.50

    # Test Dilution
    dilution = model.classify(
        "Company announces $500M secondary share dilution stock offering"
    )
    assert dilution["event_type"] == "CAPITAL_DILUTION_OFFERING"
    assert dilution["sentiment_bias"] < 0.0


def test_fast_finbert_engine():
    engine = get_fast_finbert_engine()
    res = engine.predict_single(
        "Record quarterly profits and revenue beat expectations"
    )
    assert res["sentiment_label"] in ["positive", "neutral", "negative"]
    assert "prob_positive" in res
    assert "latency_ms" in res
    assert res["confidence"] > 0.0


def test_neural_gating_network_fusion():
    gate = NeuralGatingNetwork()

    finbert_bull = {"sentiment_score": 0.85, "confidence": 0.90}
    event_bull = {
        "sentiment_bias": 0.90,
        "urgency_score": 0.85,
        "event_type": "EARNINGS_SURPRISE_UP",
        "is_material": True,
    }
    dlinear_bull = {"bullish_probability": 0.75}

    # Warmup
    _ = gate.fuse(finbert_bull, event_bull, dlinear_bull, 1.0, 1.0)

    fusion = gate.fuse(
        finbert_out=finbert_bull,
        event_out=event_bull,
        dlinear_out=dlinear_bull,
        vwap_diff_pct=1.5,
        dark_pool_ratio=1.35,
    )

    assert fusion["signal"] in ["STRONG_BUY", "BUY"]
    assert fusion["composite_score"] > 0.30
    assert sum(fusion["model_attributions"].values()) == pytest.approx(100.0, abs=1.0)
    assert fusion["latency_ms"] < 50.0


def test_bull_trap_veto_logic():
    gate = NeuralGatingNetwork()

    # Social hype but price below VWAP and whale selling
    finbert_hype = {"sentiment_score": 0.70, "confidence": 0.80}
    event_hype = {
        "sentiment_bias": 0.65,
        "urgency_score": 0.80,
        "event_type": "RETAIL_FOMO_VIRALITY",
        "is_material": False,
    }
    dlinear_lag = {"bullish_probability": 0.50}

    fusion = gate.fuse(
        finbert_out=finbert_hype,
        event_out=event_hype,
        dlinear_out=dlinear_lag,
        vwap_diff_pct=-2.5,  # Below VWAP
        dark_pool_ratio=0.75,  # Whale selling
    )

    assert fusion["veto_triggered"] is True
    # Veto should suppress signal towards NEUTRAL
    assert fusion["signal"] in ["NEUTRAL", "BUY"]
    assert fusion["composite_score"] < 0.30


def test_fast_agent_pipeline_end_to_end():
    pipeline = get_fast_agent_pipeline()
    # Warmup
    _ = pipeline.evaluate_catalyst_and_market("Warmup headline test", ticker="NVDA")

    res = pipeline.evaluate_catalyst_and_market(
        headline="FDA grants accelerated approval to breakthrough cancer therapy",
        ticker="LLY",
        vwap_diff_pct=2.0,
        dark_pool_ratio=1.40,
    )

    assert res["action"] in ["STRONG_BUY", "BUY"]
    assert res["event_type"] == "FDA_DRUG_APPROVAL"
    assert "model_attributions" in res
    assert res["total_latency_ms"] < 500.0  # Sub-second inference target

    # Verify cached sub-millisecond execution
    res_cached = pipeline.evaluate_catalyst_and_market(
        headline="FDA grants accelerated approval to breakthrough cancer therapy",
        ticker="LLY",
        vwap_diff_pct=2.0,
        dark_pool_ratio=1.40,
    )
    assert res_cached["total_latency_ms"] < 10.0
