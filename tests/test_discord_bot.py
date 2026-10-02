"""
Unit tests for src/discord_bot.py.
Verifies interactive discord bot commands (/signal, /portfolio, /execute, /var).
Uses mock brokers and isolated tmp_path to strictly protect live portfolio state.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.discord_bot import handle_bot_command, send_bot_command_reply
from src.paper_broker import PaperBroker


@pytest.fixture
def mock_isolated_broker(tmp_path):
    p_file = str(tmp_path / "mock_portfolio.json")
    t_file = str(tmp_path / "mock_trades.csv")
    broker = PaperBroker(portfolio_path=p_file, trades_path=t_file)
    broker.state = {
        "cash": 95000.0,
        "total_equity": 105000.0,
        "realized_pnl": 5000.0,
        "unrealized_pnl": 0.0,
        "total_trades": 10,
        "winning_trades": 7,
        "losing_trades": 3,
        "win_rate": 70.0,
        "open_positions": {
            "NVDA": {
                "shares": 10,
                "entry_price": 120.0,
                "entry_time": "2026-09-01T10:00:00",
                "tp1_target": 130.0,
                "sl_target": 115.0,
            }
        },
    }

    return broker


def test_handle_bot_command_help():
    res = handle_bot_command("/help")
    assert "title" in res
    assert "description" in res
    assert "Available Commands" in res["description"]


def test_handle_bot_command_var():
    with patch("src.discord_bot.run_monte_carlo_var") as mock_var:
        mock_var.return_value = {
            "var_95_dollar": 4200.0,
            "var_95_pct": 4.2,
            "cvar_95_dollar": 6100.0,
            "prob_profit_pct": 72.5,
            "worst_case_drawdown_pct": 8.5,
        }
        res = handle_bot_command("/var")
        assert "Monte Carlo" in res["title"]
        assert "$4,200.00" in res["description"]


def test_handle_bot_command_portfolio(mock_isolated_broker):
    res = handle_bot_command("/portfolio", broker=mock_isolated_broker)
    assert "Virtual Paper Portfolio" in res["title"]
    assert "NVDA" in res["description"]


def test_handle_bot_command_signal():
    with patch(
        "src.discord_bot.fetch_live_quote",
        return_value={"price": 125.0, "change_pct": 2.5},
    ):
        res = handle_bot_command("/signal NVDA")
        assert "Sentilyze Signal: NVDA" in res["title"]
        assert "AI Verdict" in res["description"]
        assert "Take-Profit" in res["description"]


def test_send_bot_command_reply_mocked():
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        success = send_bot_command_reply(
            "/signal NVDA", webhook_url="https://discord.com/api/webhooks/mock/test"
        )
        assert success is True
        assert mock_post.called
