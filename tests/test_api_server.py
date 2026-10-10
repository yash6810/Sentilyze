"""
Unit tests for FastAPI Decoupled Backend Service (src/api_server.py).
Validates all REST endpoints, portfolio inspection, session detection, and FX conversions.
Uses tmp_path for strict portfolio file preservation.
"""

import json
from fastapi.testclient import TestClient
from src.api_server import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Sentilyze Quant OS Engine API"
    assert "supported_theatres" in data
    assert "US" in data["supported_theatres"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "uptime_seconds" in data
    assert "memory_used_mb" in data
    assert "us_market_session" in data


def test_sessions_endpoint():
    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "markets" in data
    markets = data["markets"]
    assert "US" in markets
    assert "CRYPTO" in markets


def test_ticker_session_endpoint():
    # Crypto 24/7
    resp_btc = client.get("/api/v1/session/BTC-USD")
    assert resp_btc.status_code == 200
    data_btc = resp_btc.json()
    assert data_btc["can_trade_now"] is True
    assert data_btc["is_crypto"] is True

    # US Equity
    resp_us = client.get("/api/v1/session/AAPL")
    assert resp_us.status_code == 200
    data_us = resp_us.json()
    assert data_us["is_crypto"] is False
    assert "session" in data_us


def test_smartwatch_endpoint():
    resp = client.get("/api/v1/smartwatch")
    assert resp.status_code == 200
    data = resp.json()
    assert "complications" in data
    assert "circular_gauge" in data["complications"]
    assert "modular_large" in data["complications"]


def test_portfolio_endpoint_mocked(tmp_path, monkeypatch):
    mock_file = str(tmp_path / "mock_portfolio.json")
    payload = {
        "total_equity": 136545.29,
        "cash": 133052.36,
        "unrealized_pnl": 7.68,
        "realized_pnl": 54127.15,
        "win_rate": 62.5,
        "total_trades": 80,
        "open_positions": {},
        "last_updated": "2026-10-10T12:00:00Z",
    }
    with open(mock_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    import src.api_server

    monkeypatch.setattr(src.api_server, "PORTFOLIO_FILE", mock_file)

    resp = client.get("/api/v1/portfolio")
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["total_equity"] == 136545.29
    assert res_data["win_rate"] == 62.5
