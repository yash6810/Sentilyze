import pytest
from unittest.mock import MagicMock, patch
from src.alpaca_broker import AlpacaBrokerBridge


def test_alpaca_broker_initialization():
    broker = AlpacaBrokerBridge(
        api_key="TEST_KEY",
        secret_key="TEST_SECRET",
        base_url="https://paper-api.alpaca.markets",
        is_paper=True,
    )
    assert broker.api_key == "TEST_KEY"
    assert broker.secret_key == "TEST_SECRET"
    assert broker.base_url == "https://paper-api.alpaca.markets"
    assert broker.headers["APCA-API-KEY-ID"] == "TEST_KEY"


@patch("src.alpaca_broker.requests.get")
def test_alpaca_broker_is_connected(mock_get):
    mock_get.return_value.status_code = 200
    broker = AlpacaBrokerBridge(
        api_key="TEST_KEY",
        secret_key="TEST_SECRET",
        base_url="https://paper-api.alpaca.markets",
    )
    assert broker.is_connected() is True


@patch("src.alpaca_broker.requests.get")
def test_alpaca_broker_get_account_summary(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "equity": "100000.00",
        "cash": "100000.00",
        "buying_power": "400000.00",
        "currency": "USD",
        "account_number": "PA3TEST12345",
    }
    broker = AlpacaBrokerBridge(
        api_key="TEST_KEY",
        secret_key="TEST_SECRET",
        base_url="https://paper-api.alpaca.markets",
    )
    summary = broker.get_account_summary()
    assert summary["status"] == "CONNECTED"
    assert summary["equity"] == 100000.0
    assert summary["cash"] == 100000.0
    assert summary["buying_power"] == 400000.0
    assert summary["account_number"] == "PA3TEST12345"


@patch("src.alpaca_broker.requests.post")
@patch("src.alpaca_broker.requests.get")
def test_alpaca_broker_submit_bracket_order(mock_get, mock_post):
    mock_get.return_value.status_code = 200
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "id": "order-uuid-12345",
        "symbol": "NVDA",
        "qty": "10",
        "side": "buy",
        "order_class": "bracket",
    }

    broker = AlpacaBrokerBridge(
        api_key="TEST_KEY",
        secret_key="TEST_SECRET",
        base_url="https://paper-api.alpaca.markets",
    )
    res = broker.submit_bracket_order(
        ticker="NVDA", qty=10, take_profit_price=135.0, stop_loss_price=120.0
    )
    assert res["status"] == "SUBMITTED"
    assert res["order"]["id"] == "order-uuid-12345"
