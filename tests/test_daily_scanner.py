import pytest
from unittest.mock import patch, MagicMock
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
    
    # Mock features and price history
    mock_features = pd.DataFrame(
        np.ones((5, 25)),
        columns=[
            "ma7", "ma21", "sma200", "rsi", "macd", "bollinger_upper", "bollinger_lower",
            "stochastic_oscillator", "atr", "return_1d", "return_5d", "return_21d",
            "volatility_21d", "price_to_sma200", "rsi_slope", "ma_spread", "volume_ratio",
            "atr_ratio", "mean_sentiment_score", "negative", "neutral", "positive",
            "vix_close", "vix_ma5", "vix_change_1d"
        ]
    )
    mock_price_hist = pd.DataFrame({
        "Close": [100.0, 105.0, 110.0, 115.0, 120.0],
        "rsi": [55.0, 56.0, 57.0, 58.0, 60.0],
        "atr": [3.0, 3.0, 3.0, 3.0, 3.0],
        "sma200": [90.0, 91.0, 92.0, 93.0, 94.0],
    })
    mock_news = pd.DataFrame({"Title": ["Test news"]})
    
    mocker.patch(
        "src.daily_scanner.preprocess_data",
        return_value=(mock_features, mock_price_hist, mock_news),
    )
    mocker.patch(
        "src.daily_scanner.get_prediction_on_latest_data",
        return_value=(np.array([1]), np.array([[0.4, 0.6]])),
    )
    
    signals = run_daily_market_scan()
    assert len(signals) >= 1
    assert signals[0]["ticker"] in ["NVDA", "AAPL"]
    assert signals[0]["signal"] == "BUY"
    assert signals[0]["take_profit"] > signals[0]["current_price"]
    assert signals[0]["stop_loss"] < signals[0]["current_price"]
