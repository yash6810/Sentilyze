import pytest
import time
from src.working_memory_tape import WorkingMemoryTape, get_working_memory_tape


def test_working_memory_tape_insufficient_ticks():
    tape = WorkingMemoryTape(maxlen=50)
    tape.record_tick("TEST", price=100.0)
    state = tape.get_microstructure_state("TEST")
    assert state["status"] == "INSUFFICIENT_TICKS"
    assert state["tape_length"] == 1
    assert "Buffering" in tape.format_deliberation_context("TEST")


def test_working_memory_tape_bullish_absorption():
    tape = WorkingMemoryTape(maxlen=50)
    # Feed rising prices with high buying volume
    for i in range(15):
        tape.record_tick(
            "TEST",
            price=100.0 + (i * 0.2),
            volume=500.0,
            bid=99.9 + (i * 0.2),
            ask=100.1 + (i * 0.2),
        )

    state = tape.get_microstructure_state("TEST")
    assert state["status"] == "ACTIVE"
    assert state["tape_length"] == 15
    assert state["velocity_10"] > 0
    assert state["flow_toxicity_score"] > 0
    assert state["tape_pressure"] == "BULLISH_ABSORPTION"
    assert not state["spread_expansion_hazard"]


def test_working_memory_tape_bearish_dumping():
    tape = WorkingMemoryTape(maxlen=50)
    # Feed falling prices with aggressive selling volume
    for i in range(15):
        tape.record_tick(
            "TEST",
            price=100.0 - (i * 0.3),
            volume=800.0,
            bid=99.8 - (i * 0.3),
            ask=100.2 - (i * 0.3),
        )

    state = tape.get_microstructure_state("TEST")
    assert state["status"] == "ACTIVE"
    assert state["velocity_10"] < 0
    assert state["flow_toxicity_score"] < 0
    assert state["tape_pressure"] == "BEARISH_DUMPING"


def test_working_memory_tape_spread_expansion_hazard():
    tape = WorkingMemoryTape(maxlen=50)
    # Normal spread: 2 bps (bid 99.99, ask 100.01)
    for _ in range(10):
        tape.record_tick("TEST", price=100.0, volume=100.0, bid=99.99, ask=100.01)

    # Sudden spread blowout: bid 98.0, ask 102.0 (400 bps spread)
    tape.record_tick("TEST", price=100.0, volume=10.0, bid=98.0, ask=102.0)

    state = tape.get_microstructure_state("TEST")
    assert state["spread_expansion_hazard"] is True
    assert state["tape_pressure"] == "SPREAD_EXPANSION_DANGER"


def test_working_memory_tape_singleton():
    t1 = get_working_memory_tape()
    t2 = get_working_memory_tape()
    assert t1 is t2
