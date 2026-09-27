"""
Unit Tests for Overnight Futures & Macro Sentinel Engine.
STRICT PORTFOLIO PRESERVATION: Uses tmp_path and mock quotes, zero impact on results/ files.
"""

import json
import pytest
from unittest.mock import patch
from src.futures_sentinel import evaluate_overnight_futures_pulse, fetch_futures_quote


def test_evaluate_futures_bullish_gap(tmp_path):
    mock_port = {
        "total_equity": 100000.0,
        "cash": 50000.0,
        "open_positions": {
            "DAL": {
                "shares": 100,
                "current_price": 50.0,
                "entry_price": 45.0,
                "sl_target": 46.0,
            }
        },
    }
    p_file = tmp_path / "mock_portfolio.json"
    p_file.write_text(json.dumps(mock_port), encoding="utf-8")

    mock_quotes = {
        "ES=F": {"price": 5000.0, "change_pct": 0.80, "status": "LIVE"},
        "NQ=F": {"price": 18000.0, "change_pct": 1.00, "status": "LIVE"},
        "YM=F": {"price": 39000.0, "change_pct": 0.50, "status": "LIVE"},
        "^TNX": {"price": 4.25, "change_pct": -0.10, "status": "LIVE"},
        "CL=F": {"price": 75.0, "change_pct": -0.50, "status": "LIVE"},
        "^VIX": {"price": 13.5, "change_pct": -3.20, "status": "LIVE"},
    }

    with patch(
        "src.futures_sentinel.fetch_futures_quote",
        side_effect=lambda s, proxy=None: mock_quotes.get(
            s, {"price": 0.0, "change_pct": 0.0, "status": "UNAVAILABLE"}
        ),
    ):
        res = evaluate_overnight_futures_pulse(
            portfolio_path=str(p_file),
            output_path=str(tmp_path / "pulse.json"),
            dispatch_discord=False,
        )

        assert "BULLISH" in res["market_bias"]
        assert res["composite_gap_pct"] > 0.40
        assert "DAL" in res["position_impacts"]
        assert res["position_impacts"]["DAL"]["projected_dollar_impact"] > 0.0
        assert res["projected_equity_at_open"] > 100000.0


def test_evaluate_futures_bearish_gap(tmp_path):
    mock_port = {
        "total_equity": 100000.0,
        "cash": 60000.0,
        "open_positions": {
            "EMR": {
                "shares": 50,
                "current_price": 100.0,
                "entry_price": 95.0,
                "sl_target": 96.0,
            }
        },
    }
    p_file = tmp_path / "mock_portfolio.json"
    p_file.write_text(json.dumps(mock_port), encoding="utf-8")

    mock_quotes = {
        "ES=F": {"price": 4900.0, "change_pct": -0.90, "status": "LIVE"},
        "NQ=F": {"price": 17500.0, "change_pct": -1.20, "status": "LIVE"},
        "YM=F": {"price": 38000.0, "change_pct": -0.60, "status": "LIVE"},
        "^TNX": {"price": 4.50, "change_pct": 2.50, "status": "LIVE"},
        "CL=F": {"price": 85.0, "change_pct": 4.00, "status": "LIVE"},
        "^VIX": {"price": 19.5, "change_pct": 15.0, "status": "LIVE"},
    }

    with patch(
        "src.futures_sentinel.fetch_futures_quote",
        side_effect=lambda s, proxy=None: mock_quotes.get(
            s, {"price": 0.0, "change_pct": 0.0, "status": "UNAVAILABLE"}
        ),
    ):
        res = evaluate_overnight_futures_pulse(
            portfolio_path=str(p_file),
            output_path=str(tmp_path / "pulse2.json"),
            dispatch_discord=False,
        )

        assert "BEARISH" in res["market_bias"]
        assert res["composite_gap_pct"] < -0.40
        assert "EMR" in res["position_impacts"]
        assert res["position_impacts"]["EMR"]["projected_dollar_impact"] < 0.0
        assert res["projected_equity_at_open"] < 100000.0
