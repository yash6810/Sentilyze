"""
Unit test proving that when real news is missing, no order is placed by autonomous_trader.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.autonomous_trader import AutonomousTradingEngine
from src.paper_broker import PaperBroker


def test_missing_news_skips_trade_and_places_no_order(mocker):
    # Setup mock broker
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

    # Mock get_news returning an empty DataFrame with news_available=False
    empty_news_df = pd.DataFrame(
        columns=["Title", "publishedAt", "source", "url", "description"]
    )
    empty_news_df.attrs["news_available"] = False
    mocker.patch("src.autonomous_trader.get_news", return_value=empty_news_df)

    # Mock live quote
    mocker.patch(
        "src.autonomous_trader.fetch_universe_live_quotes",
        return_value={"NVDA": {"ticker": "NVDA", "price": 125.0, "status": "LIVE"}},
    )

    mock_committee = mocker.patch("src.autonomous_trader.convene_trading_committee")

    engine = AutonomousTradingEngine(broker=mock_broker)
    res = engine.run_autonomous_cycle(candidate_tickers=["NVDA"])

    # 1. Committee was never convened because missing news aborted early
    assert (
        mock_committee.call_count == 0
    ), "Trading committee should not be convened when news is missing"

    # 2. Broker never executed any buy
    assert mock_broker.execute_buy.call_count == 0
    assert mock_broker.execute_manual_buy.call_count == 0
    assert res["buys"] == []
