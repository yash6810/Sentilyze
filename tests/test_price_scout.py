import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.price_scout import (
    PriceActionScoutAgent,
    PriceScoutBot,
    get_latest_scout_alerts,
)


def test_price_scout_evaluation():
    scout = PriceActionScoutAgent()
    with (
        patch("src.price_scout.fetch_live_quote") as mock_quote,
        patch("src.price_scout.get_price_history") as mock_hist,
    ):
        mock_quote.return_value = {
            "ticker": "NVDA",
            "price": 130.0,
            "prev_close": 125.0,
            "day_high": 131.0,
            "day_low": 124.0,
            "change_pct": 4.0,
        }
        dates = pd.date_range("2026-01-01", periods=10)
        mock_hist.return_value = pd.DataFrame(
            {
                "Close": [120.0 + i for i in range(10)],
                "Volume": [1000000 + i * 10000 for i in range(10)],
            },
            index=dates,
        )

        res = scout.evaluate("NVDA", spot_price=130.0)
        assert res["ticker"] == "NVDA"
        assert res["spot_price"] == 130.0
        assert res["vote"] in ["BUY", "HOLD", "SELL"]
        assert 0.0 <= res["conviction_score"] <= 100.0
        assert "range_position_pct" in res
        assert "rvol" in res


def test_price_scout_bot_scan():
    bot = PriceScoutBot()
    with patch.object(
        PriceActionScoutAgent,
        "evaluate",
        return_value={
            "ticker": "AAPL",
            "vote": "BUY",
            "conviction_score": 78.0,
            "spot_price": 220.0,
        },
    ):
        picks = bot.scan_universe_breakouts(["AAPL", "MSFT"])
        assert len(picks) > 0
        assert picks[0]["ticker"] == "AAPL"
        assert picks[0]["conviction_score"] == 78.0

    alerts = get_latest_scout_alerts()
    assert "candidates" in alerts
