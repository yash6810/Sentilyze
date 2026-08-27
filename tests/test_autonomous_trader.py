import pytest
from unittest.mock import MagicMock, patch
from src.autonomous_trader import AutonomousTradingEngine, load_universe_tickers
from src.paper_broker import PaperBroker


def test_load_universe_tickers():
    tickers = load_universe_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) >= 5
    assert "NVDA" in tickers or "AAPL" in tickers


@patch("src.autonomous_trader.fetch_universe_live_quotes")
@patch("src.autonomous_trader.get_news")
@patch("src.autonomous_trader.convene_trading_committee")
def test_autonomous_cycle_execution(mock_committee, mock_news, mock_quotes):
    mock_quotes.return_value = {
        "NVDA": {"ticker": "NVDA", "price": 125.0, "status": "LIVE"},
        "AAPL": {"ticker": "AAPL", "price": 220.0, "status": "LIVE"},
    }
    mock_news.return_value = MagicMock()
    mock_committee.return_value = {
        "ticker": "NVDA",
        "spot_price": 125.0,
        "final_resolution": "🚀 CONVICTION INSTITUTIONAL BUY",
        "action_code": "EXECUTE_BUY",
        "consensus_conviction_pct": 85.0,
        "tp1_target": 132.0,
        "tp2_target": 140.0,
        "stop_loss_target": 119.0,
        "cro_signoff": {
            "approved_leverage": 1.5,
            "kelly_allocation_pct": 12.5,
            "consensus_conviction_pct": 85.0,
        },
    }

    mock_broker = MagicMock(spec=PaperBroker)
    mock_broker.state = {
        "cash": 100000.0,
        "open_positions": {},
        "closed_trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_equity": 100000.0,
    }
    mock_broker.get_portfolio_summary.return_value = {
        "total_equity": 100000.0,
        "cash": 100000.0,
        "unrealized_pnl": 0.0,
    }
    mock_broker.execute_manual_buy.return_value = {
        "success": True,
        "shares": 100,
        "price": 125.0,
    }

    engine = AutonomousTradingEngine(broker=mock_broker)
    res = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])

    assert isinstance(res, dict)
    assert "buys" in res
    assert "portfolio_equity" in res
    assert res["portfolio_equity"] == 100000.0
