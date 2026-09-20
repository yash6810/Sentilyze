import pytest
from src.voice_desk import answer_voice_inquiry


def test_voice_inquiry_cash_buffer():
    res = answer_voice_inquiry("What is our current cash buffer and total equity?")
    assert res["category"] == "PORTFOLIO_CAPITAL"
    assert "total equity" in res["spoken_response"]
    assert "cash buffer" in res["spoken_response"]


def test_voice_inquiry_macro():
    res = answer_voice_inquiry("When is the next FOMC interest rate meeting?")
    assert res["category"] == "MACRO_CALENDAR"
    assert "Macro" in res["spoken_response"] or "FOMC" in res["spoken_response"]


def test_voice_inquiry_microstructure():
    res = answer_voice_inquiry("What is the VPIN order flow toxicity for NVDA?")
    assert res["category"] == "MICROSTRUCTURE"
    assert "VPIN" in res["spoken_response"]
    assert "NVDA" in res["spoken_response"]


def test_voice_inquiry_committee():
    res = answer_voice_inquiry(
        "How did the committee vote on NVDA?", selected_ticker="NVDA"
    )
    assert res["category"] == "COMMITTEE_VERDICT"
    assert "Chief Risk Officer" in res["spoken_response"]
    assert "NVDA" in res["spoken_response"]
