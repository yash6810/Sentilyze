"""
Unit tests (mocked, zero-network) for Alpaca connectivity diagnostic logging,
funnel honesty reporting, and CI pre-flight workflow checks.
"""

import json
import logging
from unittest.mock import MagicMock, patch
import pytest

from src.alpaca_broker import AlpacaBrokerBridge
from src.funnel_tracker import record_funnel_run
from scripts.check_alpaca_connectivity import check_alpaca_connectivity


def test_alpaca_warning_logging_missing_keys(caplog):
    """Item 1: Test WARNING logged when keys are missing without exposing secrets."""
    caplog.set_level(logging.WARNING)
    broker = AlpacaBrokerBridge(
        api_key="",
        secret_key="",
        base_url="https://paper-api.alpaca.markets",
    )
    is_conn = broker.is_connected()
    assert is_conn is False

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Alpaca connectivity check warning" in w.message for w in warnings)
    warning_text = warnings[-1].message
    assert "key_env_vars_present" in warning_text
    assert "host='paper-api.alpaca.markets'" in warning_text
    assert "status_code=None" in warning_text
    assert "Missing API key or Secret key" in warning_text


@patch("src.alpaca_broker.requests.get")
def test_alpaca_warning_logging_http_error(mock_get, caplog):
    """Item 1: Test WARNING logged on non-200 HTTP response with truncated body and no leaked secrets."""
    caplog.set_level(logging.WARNING)
    fake_key = "SECRET_KEY_NEVER_LEAK_12345"
    fake_secret = "SECRET_SECRET_NEVER_LEAK_67890"

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden: Invalid API key credentials" * 10
    mock_get.return_value = mock_resp

    broker = AlpacaBrokerBridge(
        api_key=fake_key,
        secret_key=fake_secret,
        base_url="https://paper-api.alpaca.markets",
    )
    is_conn = broker.is_connected()
    assert is_conn is False

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Alpaca connectivity check warning" in w.message for w in warnings)
    warning_text = warnings[-1].message
    assert "status_code=403" in warning_text
    assert "host='paper-api.alpaca.markets'" in warning_text

    # Strict security check: secrets and raw headers must never appear in log records
    for record in caplog.records:
        assert fake_key not in record.message
        assert fake_secret not in record.message
        assert "APCA-API-KEY-ID" not in record.message


@patch("src.alpaca_broker.requests.get")
def test_alpaca_warning_get_account_summary_disconnected(mock_get, caplog):
    """Item 1: Test get_account_summary logs warning and returns fallback when broker is disconnected."""
    caplog.set_level(logging.WARNING)
    broker = AlpacaBrokerBridge(
        api_key="",
        secret_key="",
        base_url="https://paper-api.alpaca.markets",
    )
    summary = broker.get_account_summary()
    assert summary["status"] == "DISCONNECTED"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("AlpacaBrokerBridge.get_account_summary" in w.message for w in warnings)


def test_funnel_honesty_aborted_run(tmp_path):
    """Item 2: When live run aborts due to unavailable Alpaca, record runs_aborted_alpaca_unavailable=1, tickers_scanned=0, total_buy_blocked=0."""
    json_file = tmp_path / "funnel.json"
    md_file = tmp_path / "funnel.md"
    log_file = tmp_path / "funnel_log.jsonl"

    rec = record_funnel_run(
        tickers_scanned=0,
        skipped_no_real_news=0,
        vetoed_by_reason={},
        buy_blocked_guards={},
        orders_submitted=0,
        skipped_news_quota_or_error=0,
        runs_aborted_alpaca_unavailable=1,
        json_path=str(json_file),
        md_path=str(md_file),
        log_path=str(log_file),
    )

    assert rec["runs_aborted_alpaca_unavailable"] == 1
    assert rec["tickers_scanned"] == 0
    assert rec["total_buy_blocked"] == 0
    assert rec["orders_submitted"] == 0

    # Verify JSON file
    with open(json_file, "r", encoding="utf-8") as jf:
        data = json.load(jf)
    assert data["runs_aborted_alpaca_unavailable"] == 1
    assert data["tickers_scanned"] == 0
    assert data["total_buy_blocked"] == 0

    # Verify Markdown file contains Total Runs and Aborted Runs
    with open(md_file, "r", encoding="utf-8") as mf:
        content = mf.read()
    assert "| Total Runs | Aborted Runs | Scanned |" in content
    assert "| 1 | 1 | 0 | 0 | 0 | 0 | 0 | **0** |" in content


@patch("src.market_session.get_us_market_session")
def test_autonomous_trader_live_alpaca_unavailable_records_aborted_run(
    mock_market_session, tmp_path, monkeypatch
):
    """Item 2: Autonomous trader live mode abort records tickers_scanned=0 and runs_aborted_alpaca_unavailable=1."""
    mock_market_session.return_value = {
        "status": "REGULAR_MARKET_OPEN",
        "is_open": True,
        "time_edt": "10:00:00 AM EDT",
        "utc_time": "2026-10-27 14:00:00 UTC",
    }
    from src.autonomous_trader import AutonomousTradingEngine

    monkeypatch.setenv("LIVE_ALPACA_REQUIRED", "true")
    monkeypatch.setenv("FUNNEL_DIR", str(tmp_path))

    from src.paper_broker import PaperBroker

    mock_broker = PaperBroker(
        portfolio_file=str(tmp_path / "mock_portfolio.json"),
        trades_file=str(tmp_path / "mock_trades.csv"),
    )
    mock_crypto = PaperBroker(
        portfolio_file=str(tmp_path / "mock_crypto_portfolio.json"),
        trades_file=str(tmp_path / "mock_crypto_trades.csv"),
    )

    engine = AutonomousTradingEngine(broker=mock_broker, crypto_broker=mock_crypto)
    mock_alpaca = MagicMock()
    mock_alpaca.is_connected.return_value = False
    engine.alpaca = mock_alpaca

    actions = engine._execute_cycle_body(candidate_tickers=["NVDA", "AAPL", "MSFT"])
    assert actions.get("buys") == []

    # Read written funnel.json
    funnel_json_path = tmp_path / "funnel.json"
    assert funnel_json_path.exists()
    with open(funnel_json_path, "r", encoding="utf-8") as f:
        f_data = json.load(f)

    assert f_data["runs_aborted_alpaca_unavailable"] == 1
    assert f_data["tickers_scanned"] == 0
    assert f_data["total_buy_blocked"] == 0
    assert f_data["buy_blocked_guards"] == {}


@patch("scripts.check_alpaca_connectivity.requests.get")
def test_check_alpaca_connectivity_success(mock_get, capsys, monkeypatch):
    """Item 3: Pre-flight script exits 0 on HTTP 200, prints only status and boolean flags."""
    fake_key = "SUPER_SECRET_KEY_ABC"
    fake_secret = "SUPER_SECRET_SECRET_XYZ"
    monkeypatch.setenv("ALPACA_API_KEY", fake_key)
    monkeypatch.setenv("ALPACA_SECRET_KEY", fake_secret)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp

    exit_code = check_alpaca_connectivity()
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "api_key_present=True" in captured
    assert "secret_key_present=True" in captured
    assert "status_code=200" in captured
    assert fake_key not in captured
    assert fake_secret not in captured


@patch("scripts.check_alpaca_connectivity.requests.get")
def test_check_alpaca_connectivity_failure_status(mock_get, capsys, monkeypatch):
    """Item 3: Pre-flight script exits 1 on non-200 HTTP status."""
    monkeypatch.setenv("ALPACA_API_KEY", "SOME_KEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "SOME_SECRET")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_get.return_value = mock_resp

    exit_code = check_alpaca_connectivity()
    assert exit_code == 1

    captured = capsys.readouterr().out
    assert "api_key_present=True" in captured
    assert "secret_key_present=True" in captured
    assert "status_code=401" in captured


@patch(
    "scripts.check_alpaca_connectivity.requests.get",
    side_effect=Exception("Connection refused"),
)
def test_check_alpaca_connectivity_exception(mock_get, capsys, monkeypatch):
    """Item 3: Pre-flight script exits 1 on network exception."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)

    exit_code = check_alpaca_connectivity()
    assert exit_code == 1

    captured = capsys.readouterr().out
    assert "api_key_present=False" in captured
    assert "secret_key_present=False" in captured
    assert "status_code=None" in captured
