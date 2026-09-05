from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from src.daily_scanner import run_daily_market_scan


def test_run_daily_market_scan(mocker, tmp_path):
    """
    Test that run_daily_market_scan executes successfully across mock tickers.
    """
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="NVDA\nAAPL\n"),
    )

    mock_model = MagicMock()
    mocker.patch("src.daily_scanner.load_model", return_value=mock_model)

    from src.config import FEATURES

    # Mock features and price history
    mock_features = pd.DataFrame(
        np.ones((5, len(FEATURES))),
        columns=FEATURES,
    )
    mock_price_hist = pd.DataFrame(
        {
            "Open": [99.0, 104.0, 109.0, 114.0, 119.0],
            "High": [101.0, 106.0, 111.0, 116.0, 121.0],
            "Low": [98.0, 103.0, 108.0, 113.0, 118.0],
            "Close": [100.0, 105.0, 110.0, 115.0, 120.0],
            "Volume": [1000, 1200, 1500, 1800, 2000],
            "rsi": [55.0, 56.0, 57.0, 58.0, 60.0],
            "atr": [3.0, 3.0, 3.0, 3.0, 3.0],
            "sma200": [90.0, 91.0, 92.0, 93.0, 94.0],
        }
    )
    mock_news = pd.DataFrame({"Title": ["Test news"]})

    mocker.patch(
        "src.daily_scanner.preprocess_data",
        return_value=(mock_features, mock_price_hist, mock_news),
    )
    mocker.patch(
        "src.daily_scanner.get_prediction_on_latest_data",
        return_value=(np.array([1]), np.array([[0.4, 0.6]])),
    )

    mock_broker = MagicMock()
    mock_broker.get_portfolio_summary.return_value = {
        "total_equity": 100000.0,
        "cash": 100000.0,
        "open_positions_count": 0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
    }
    mock_broker.state = {"open_positions": {}, "closed_trades": []}
    mocker.patch("src.paper_broker.PaperBroker", return_value=mock_broker)

    signals = run_daily_market_scan()
    assert len(signals) >= 1
    assert signals[0]["ticker"] in ["NVDA", "AAPL"]
    assert signals[0]["signal"] == "BUY"
    assert signals[0]["take_profit"] > signals[0]["current_price"]
    assert signals[0]["stop_loss"] < signals[0]["current_price"]
