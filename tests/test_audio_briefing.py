"""
Unit tests for Voice Morning Audio Briefing (Sprint 1, Module 1.10)
"""

import pytest
from src.audio_briefing import generate_audio_script


def test_generate_audio_script_with_buys():
    signals = [
        {"ticker": "NVDA", "signal": "BUY", "confidence": 0.85},
        {"ticker": "MSFT", "signal": "BUY", "confidence": 0.78},
        {"ticker": "AAPL", "signal": "HOLD", "confidence": 0.50},
    ]
    script = generate_audio_script(signals, total_equity=145000.0)
    assert "NVDA" in script
    assert "145,000.00" in script
    assert "Sentilyze Autonomous AI Market Intelligence Briefing" in script


def test_generate_audio_script_empty():
    script = generate_audio_script([], total_equity=100000.0)
    assert "neutral or defensive mode" in script
    assert "100,000.00" in script
