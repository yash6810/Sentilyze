"""
Unit tests for Instant Trade Telemetry & Results Engine.
"""

from src.trade_telemetry import get_trade_telemetry, print_formatted_trade_report


def test_get_trade_telemetry_structure():
    """Verify telemetry returns all required reporting keys."""
    data = get_trade_telemetry()
    assert isinstance(data, dict)
    assert "total_equity" in data
    assert "cash" in data
    assert "realized_pnl" in data
    assert "unrealized_pnl" in data
    assert "win_rate_pct" in data
    assert "today" in data
    assert "yesterday" in data
    assert "trades" in data["today"]
    assert "trades" in data["yesterday"]
    assert "recent_closed_trades" in data
    assert data["total_equity"] >= 100000.0


def test_print_formatted_trade_report_runs(capsys):
    """Verify formatted trade report prints without raising exceptions."""
    print_formatted_trade_report()
    captured = capsys.readouterr()
    assert "SENTILYZE INSTITUTIONAL PORTFOLIO & TRADE RESULTS" in captured.out
    assert "Total Equity" in captured.out
    assert "Cash Balance" in captured.out
