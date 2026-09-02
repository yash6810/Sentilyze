from unittest.mock import patch, MagicMock
from src.alerts import (
    format_signal_card,
    send_discord_alert,
    send_discord_execution_alert,
    send_discord_committee_alert,
    send_discord_social_spike_alert,
    send_discord_market_pulse,
    send_discord_digest,
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
def test_send_discord_execution_alert(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_post.return_value = mock_response

    # Buy Entry
    res_buy = send_discord_execution_alert(
        {
            "action": "BUY",
            "stage": "ENTRY",
            "ticker": "NVDA",
            "price": 130.0,
            "shares": 50,
            "kelly_pct": 10.0,
        },
        webhook_url="https://discord.com/api/webhooks/mock",
    )
    assert res_buy is True

    # TP1 Exit
    res_tp1 = send_discord_execution_alert(
        {
            "action": "SELL",
            "stage": "TP1_PROFIT_LOCK",
            "ticker": "NVDA",
            "price": 138.0,
            "shares": 25,
            "realized_pnl": 200.0,
        },
        webhook_url="https://discord.com/api/webhooks/mock",
    )
    assert res_tp1 is True


@patch("requests.post")
def test_send_discord_committee_and_pulse(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    delib = {
        "ticker": "NVDA",
        "final_verdict": "CONVICTION_BUY",
        "cro_signoff": {
            "status": "APPROVED",
            "macro_vix_level": 15.0,
            "approved_kelly_pct": 8.0,
        },
        "committee_votes": {},
    }
    assert (
        send_discord_committee_alert(
            delib, webhook_url="https://discord.com/api/webhooks/mock"
        )
        is True
    )

    soc = {
        "ticker": "NVDA",
        "mention_velocity_ratio": 2.8,
        "bullish_sentiment_pct": 80.0,
        "regime": "VIRAL",
    }
    assert (
        send_discord_social_spike_alert(
            soc, webhook_url="https://discord.com/api/webhooks/mock"
        )
        is True
    )

    pulse = {
        "vix_level": 14.5,
        "vix_regime": "BULL",
        "top_buys": [{"ticker": "NVDA", "price": 130.0, "confidence": 0.85}],
    }
    assert (
        send_discord_market_pulse(
            pulse, webhook_url="https://discord.com/api/webhooks/mock"
        )
        is True
    )

    digest = [
        {
            "ticker": "NVDA",
            "signal": "BUY",
            "confidence": 0.85,
            "current_price": 130.0,
            "stop_loss": 125.0,
        }
    ]
    assert (
        send_discord_digest(digest, webhook_url="https://discord.com/api/webhooks/mock")
        is True
    )


@patch("requests.post")
def test_send_discord_premarket_briefing(mock_post):
    from src.alerts import send_discord_premarket_briefing

    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_post.return_value = mock_response

    summary = {
        "total_equity": 105000.0,
        "cash": 95000.0,
        "unrealized_pnl": 500.0,
        "unrealized_pnl_pct": 0.5,
        "win_rate": 60.0,
        "open_positions": {"NVDA": {"shares": 10}},
    }
    top_watchlist = [
        {
            "ticker": "NVDA",
            "resolution": "BUY",
            "conviction": 85.0,
            "sentiment_score": 0.5,
        }
    ]

    success = send_discord_premarket_briefing(
        portfolio_summary=summary,
        macro_vix=17.5,
        top_watchlist=top_watchlist,
        webhook_url="https://discord.com/api/webhooks/mock",
    )
    assert success is True
    assert mock_post.called
