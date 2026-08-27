from unittest.mock import patch, MagicMock
from src.alerts import (
    format_signal_card,
    send_discord_alert,
    send_telegram_alert,
)


def test_format_signal_card():
    payload = format_signal_card(
        ticker="NVDA",
        signal="BUY",
        confidence=0.85,
        current_price=125.50,
        stop_loss=120.00,
        regime="BULLISH",
        top_features=[{"feature": "return_5d", "importance": 0.4}],
    )
    assert payload["ticker"] == "NVDA"
    assert payload["signal"] == "BUY"
    assert payload["confidence"] == 0.85
    assert payload["current_price"] == 125.50
    assert payload["stop_loss"] == 120.00
    assert "timestamp" in payload


@patch("requests.post")
def test_send_discord_alert(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_post.return_value = mock_response

    payload = format_signal_card(
        ticker="AAPL",
        signal="BUY",
        confidence=0.80,
        current_price=220.0,
        stop_loss=215.0,
        regime="BULLISH",
        top_features=[],
    )

    success = send_discord_alert(
        payload, webhook_url="https://discord.com/api/webhooks/mock"
    )
    assert success is True
    assert mock_post.called


@patch("requests.post")
def test_send_telegram_alert(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    payload = format_signal_card(
        ticker="MSFT",
        signal="SELL",
        confidence=0.75,
        current_price=410.0,
        stop_loss=420.0,
        regime="BEARISH",
        top_features=[],
    )

    success = send_telegram_alert(
        payload, bot_token="12345:mock_token", chat_id="98765"
    )
    assert success is True
    assert mock_post.called
