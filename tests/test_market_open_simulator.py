"""
Unit Tests for Market-Open Order Execution Simulator.
STRICT PORTFOLIO PRESERVATION: Uses tmp_path and mock files, zero impact on results/ files.
"""

import json
import pytest
from src.market_open_simulator import simulate_market_opening


def test_market_open_simulation_execution(tmp_path):
    # Setup mock portfolio
    mock_port = {
        "total_equity": 100000.0,
        "cash": 80000.0,
        "open_positions": {
            "DAL": {
                "shares": 100,
                "entry_price": 50.0,
                "current_price": 52.0,
                "sl_target": 51.0,
                "tp1_target": 55.0,
            }
        },
    }
    p_file = tmp_path / "paper_portfolio.json"
    p_file.write_text(json.dumps(mock_port), encoding="utf-8")

    # Setup mock watchlist
    mock_wl = {
        "watchlist": [
            {
                "ticker": "MSFT",
                "conviction_pct": 82.5,
                "resolution": "HIGH CONVICTION COMMITTEE BUY",
            }
        ]
    }
    wl_file = tmp_path / "watchlist.json"
    wl_file.write_text(json.dumps(mock_wl), encoding="utf-8")

    # Setup mock futures pulse
    mock_pulse = {
        "composite_gap_pct": 0.65,
        "market_bias": "BULLISH_GAP_OPEN",
    }
    pulse_file = tmp_path / "futures_pulse.json"
    pulse_file.write_text(json.dumps(mock_pulse), encoding="utf-8")

    out_file = tmp_path / "plan.json"

    res = simulate_market_opening(
        portfolio_path=str(p_file),
        watchlist_path=str(wl_file),
        futures_pulse_path=str(pulse_file),
        output_path=str(out_file),
    )

    assert "active_position_routing" in res
    assert "opening_order_ticket" in res

    # Position routing asserts
    expected_scen = res["active_position_routing"]["expected_futures_gap"]["DAL"]
    assert expected_scen["action"] == "HOLD_AND_TRAIL"
    assert expected_scen["simulated_open_price"] > 52.0

    # Order ticket asserts
    ticket = res["opening_order_ticket"]
    assert ticket["candidate"] == "MSFT"
    assert ticket["conviction_pct"] == 82.5
    assert ticket["total_shares"] > 0
    assert len(ticket["almgren_chriss_execution"]["slices"]) == 10
    assert out_file.exists()
