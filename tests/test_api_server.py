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
    assert "global_sessions" in data


def test_sessions_endpoint():
    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "markets" in data
    markets = data["markets"]
    assert "US" in markets
    assert "CRYPTO" in markets
    assert "JAPAN" in markets
    assert "INDIA" in markets
    assert "HONG_KONG" in markets


def test_ticker_session_endpoint():
    # Crypto 24/7
    resp_btc = client.get("/api/v1/session/BTC-USD")
    assert resp_btc.status_code == 200
    data_btc = resp_btc.json()
    assert data_btc["can_trade_now"] is True
    assert data_btc["session"]["market_code"] == "CRYPTO"

    # Japan TSE
    resp_tse = client.get("/api/v1/session/7203.T")
    assert resp_tse.status_code == 200
    data_tse = resp_tse.json()
    assert data_tse["session"]["market_code"] == "JP"
    assert data_tse["session"]["currency"] == "JPY"

    # India NSE
    resp_nse = client.get("/api/v1/session/RELIANCE.NS")
    assert resp_nse.status_code == 200
    data_nse = resp_nse.json()
    assert data_nse["session"]["market_code"] == "IN"
    assert data_nse["session"]["currency"] == "INR"


def test_fx_endpoint():
    resp = client.get("/api/v1/fx/JPY?amount=10000")
    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "JPY"
    assert data["amount_local"] == 10000.0
    assert data["amount_usd"] > 0


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
