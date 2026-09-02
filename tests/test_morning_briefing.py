import pytest
from src.morning_briefing import (
    generate_morning_briefing_text,
    synthesize_briefing_audio,
)


def test_generate_morning_briefing_text():
    memo = generate_morning_briefing_text("NVDA")
    assert "headline" in memo
    assert "executive_summary" in memo
    assert "audio_script" in memo
    assert len(memo["audio_script"]) > 50
    assert "macro_posture" in memo
    assert "portfolio_status" in memo


def test_synthesize_briefing_audio_execution():
    test_script = (
        "Good morning. This is your Sentilyze quantitative morning test briefing."
    )
    path = synthesize_briefing_audio(
        test_script, output_path="results/test_briefing.mp3"
    )
    assert path is not None
