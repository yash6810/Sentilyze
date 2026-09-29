"""
Unit Tests for Weekend Institutional Additions:
1. Macro Calendar & Earnings Radar
2. 10,000-Path Monte Carlo Stress Test
3. Executive Weekly Tear-Sheet
4. Deep Trade Post-Mortem & Rule Inducer
STRICT PORTFOLIO PRESERVATION: Uses tmp_path and synthetic fixtures, zero impact on results/ files.
"""

import os
import json
import pytest
import pandas as pd
from src.weekly_calendar_radar import build_weekly_radar
from src.portfolio_monte_carlo import run_portfolio_monte_carlo_simulation
from src.weekly_tearsheet import generate_weekly_executive_tearsheet
from src.trade_memory_autopsy import run_trade_memory_autopsy


def test_weekly_radar_generation(tmp_path):
    p_file = tmp_path / "mock_portfolio.json"
    p_file.write_text(
        json.dumps({"open_positions": {"DAL": {}, "EMR": {}}}), encoding="utf-8"
    )
    w_file = tmp_path / "mock_watchlist.json"
    w_file.write_text(json.dumps({"watchlist": [{"ticker": "MSFT"}]}), encoding="utf-8")
    out_file = tmp_path / "out_radar.json"

    res = build_weekly_radar(
        portfolio_path=str(p_file),
        watchlist_path=str(w_file),
        output_path=str(out_file),
    )
    assert "high_impact_macro_events" in res
    assert "monitored_earnings" in res
    assert len(res["high_impact_macro_events"]) > 0
    assert out_file.exists()


def test_portfolio_monte_carlo_simulation(tmp_path):
    p_file = tmp_path / "mock_portfolio.json"
    p_file.write_text(
        json.dumps(
            {
                "total_equity": 100000.0,
                "cash": 70000.0,
                "open_positions": {
                    "DAL": {"shares": 50, "current_price": 50.0, "entry_price": 45.0},
                    "EMR": {"shares": 50, "current_price": 100.0, "entry_price": 95.0},
                },
            }
        ),
        encoding="utf-8",
    )
    out_file = tmp_path / "out_mc.json"

    res = run_portfolio_monte_carlo_simulation(
        portfolio_path=str(p_file),
        num_simulations=500,
        forecast_days=3,
        cppi_floor=95000.0,
        output_path=str(out_file),
    )
    assert res["num_simulations"] == 500
    assert "risk_metrics" in res
    assert "var_95_dollar" in res["risk_metrics"]
    assert "terminal_equity_percentiles" in res
    assert out_file.exists()


def test_weekly_executive_tearsheet(tmp_path):
    p_file = tmp_path / "mock_portfolio.json"
    p_file.write_text(
        json.dumps(
            {
                "total_equity": 120000.0,
                "cash": 80000.0,
                "realized_pnl": 20000.0,
                "win_rate": 70.0,
                "total_trades": 10,
                "open_positions": {
                    "EMR": {
                        "shares": 20,
                        "entry_price": 100.0,
                        "current_price": 105.0,
                        "sl_target": 102.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    t_file = tmp_path / "mock_trades.csv"
    df = pd.DataFrame(
        [
            {
                "ticker": "PLTR",
                "pnl": 5000.0,
                "return_pct": 50.0,
                "reason": "TP2_RUNNER",
            },
            {"ticker": "AMD", "pnl": -200.0, "return_pct": -2.0, "reason": "STOP_LOSS"},
        ]
    )
    df.to_csv(t_file, index=False)
    out_html = tmp_path / "out_tearsheet.html"

    html = generate_weekly_executive_tearsheet(
        portfolio_path=str(p_file),
        trades_path=str(t_file),
        output_html_path=str(out_html),
    )
    assert "Sentilyze Quantitative Desk Tear-Sheet" in html
    assert "PLTR" in html
    assert out_html.exists()


def test_trade_memory_autopsy(tmp_path):
    t_file = tmp_path / "mock_trades.csv"
    df = pd.DataFrame(
        [
            {
                "ticker": "PLTR",
                "pnl": 8000.0,
                "return_pct": 490.0,
                "reason": "TP2_RUNNER_EXIT",
            },
            {"ticker": "GEV", "pnl": 500.0, "return_pct": 5.0, "reason": "MODEL_SELL"},
            {"ticker": "DAL", "pnl": -50.0, "return_pct": -0.8, "reason": "STOP_LOSS"},
            {
                "ticker": "PGR",
                "pnl": -1400.0,
                "return_pct": -3.3,
                "reason": "STOP_LOSS",
            },
        ]
    )
    df.to_csv(t_file, index=False)
    out_file = tmp_path / "out_autopsy.json"

    res = run_trade_memory_autopsy(
        trades_path=str(t_file),
        output_summary_path=str(out_file),
    )
    assert res["total_trades_analyzed"] == 4
    assert res["win_rate_pct"] == 50.0
    assert len(res["induced_cognitive_rules"]) >= 4
    assert res["archetype_distribution"]["compounding_runners_count"] == 1
    assert out_file.exists()
