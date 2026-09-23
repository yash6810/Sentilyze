import os
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from src.daily_scanner import run_daily_market_scan


def test_run_daily_market_scan_institutional_buy(mocker, tmp_path):
    """
    Test that run_daily_market_scan produces BUY when all institutional
    gates (Conviction >= 0.60, RS > SPY, VPIN < 0.70, Council pass) are met.
    """
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="NVDA\nAAPL\n"),
    )

    mock_model = MagicMock()
    mocker.patch("src.daily_scanner.load_model", return_value=mock_model)

    from src.config import FEATURES

    mock_features = pd.DataFrame(
        np.ones((25, len(FEATURES))),
        columns=FEATURES,
    )
    # Upward trending price history (outperforming)
    close_vals = [100.0 + (i * 2.0) for i in range(25)]
    mock_price_hist = pd.DataFrame(
        {
            "Open": [c - 1.0 for c in close_vals],
            "High": [c + 1.0 for c in close_vals],
            "Low": [c - 2.0 for c in close_vals],
            "Close": close_vals,
            "Volume": [1000 + i * 50 for i in range(25)],
            "rsi": [55.0] * 25,
            "atr": [3.0] * 25,
            "sma200": [90.0] * 25,
        }
    )
    mock_news = pd.DataFrame({"Title": ["Test news"]})

    mocker.patch(
        "src.daily_scanner.preprocess_data",
        return_value=(mock_features, mock_price_hist, mock_news),
    )
    mocker.patch(
        "src.daily_scanner.get_prediction_on_latest_data",
        return_value=(np.array([1]), np.array([[0.35, 0.65]])),
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

    # Mock SPY history
    mock_spy = pd.DataFrame(
        {
            "Close": [500.0 + i for i in range(25)],
        }
    )
    mocker.patch("src.data_ingestion.get_price_history", return_value=mock_spy)

    # Mock VPIN
    mocker.patch(
        "src.vpin_toxicity.evaluate_ticker_toxicity",
        return_value={
            "vpin": 0.45,
            "cro_veto_recommended": False,
            "is_toxic_dumping": False,
        },
    )

    # Mock Correlation Shield
    mocker.patch(
        "src.correlation_shield.check_correlation_shield",
        return_value={"allowed": True, "reason": "Diversified"},
    )

    # Mock Committee
    mocker.patch(
        "src.agent_committee.convene_trading_committee",
        return_value={
            "consensus_direction": "BUY",
            "conviction_score": 0.72,
            "veto_active": False,
        },
    )

    signals = run_daily_market_scan()
    assert len(signals) >= 1
    assert signals[0]["ticker"] in ["NVDA", "AAPL"]
    assert signals[0]["signal"] == "BUY"
    assert signals[0]["take_profit"] > signals[0]["current_price"]
    assert signals[0]["stop_loss"] < signals[0]["current_price"]


def test_vpin_toxicity_rejects_entry(mocker):
    """
    Test that high VPIN toxicity vetoes a trade and downgrades signal to HOLD.
    """
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data="NVDA\n"))
    mocker.patch("src.daily_scanner.load_model", return_value=MagicMock())

    from src.config import FEATURES

    mock_features = pd.DataFrame(np.ones((25, len(FEATURES))), columns=FEATURES)
    close_vals = [100.0 + i for i in range(25)]
    mock_price_hist = pd.DataFrame(
        {
            "Open": close_vals,
            "High": close_vals,
            "Low": close_vals,
            "Close": close_vals,
            "Volume": [1000] * 25,
            "rsi": [55.0] * 25,
            "atr": [3.0] * 25,
            "sma200": [90.0] * 25,
        }
    )

    mocker.patch(
        "src.daily_scanner.preprocess_data",
        return_value=(mock_features, mock_price_hist, pd.DataFrame()),
    )
    mocker.patch(
        "src.daily_scanner.get_prediction_on_latest_data",
        return_value=(np.array([1]), np.array([[0.30, 0.70]])),
    )

    # VPIN Toxic offloading
    mocker.patch(
        "src.vpin_toxicity.evaluate_ticker_toxicity",
        return_value={
            "vpin": 0.82,
            "cro_veto_recommended": True,
            "is_toxic_dumping": True,
        },
    )
    mocker.patch(
        "src.paper_broker.PaperBroker",
        return_value=MagicMock(state={"open_positions": {}}),
    )

    signals = run_daily_market_scan()
    assert len(signals) >= 1
    assert signals[0]["signal"] == "HOLD"


def test_severe_breakdown_signals_sell(mocker):
    """
    Test that confidence < 0.40 triggers explicit SELL signal.
    """
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data="NVDA\n"))
    mocker.patch("src.daily_scanner.load_model", return_value=MagicMock())

    from src.config import FEATURES

    mock_features = pd.DataFrame(np.ones((25, len(FEATURES))), columns=FEATURES)
    close_vals = [100.0 - i for i in range(25)]
    mock_price_hist = pd.DataFrame(
        {
            "Open": close_vals,
            "High": close_vals,
            "Low": close_vals,
            "Close": close_vals,
            "Volume": [1000] * 25,
            "rsi": [35.0] * 25,
            "atr": [3.0] * 25,
            "sma200": [120.0] * 25,
        }
    )

    mocker.patch(
        "src.daily_scanner.preprocess_data",
        return_value=(mock_features, mock_price_hist, pd.DataFrame()),
    )
    # Low confidence 0.30 (< 0.40)
    mocker.patch(
        "src.daily_scanner.get_prediction_on_latest_data",
        return_value=(np.array([0]), np.array([[0.70, 0.30]])),
    )
    mocker.patch(
        "src.paper_broker.PaperBroker",
        return_value=MagicMock(state={"open_positions": {}}),
    )

    signals = run_daily_market_scan()
    assert len(signals) >= 1
    assert signals[0]["signal"] == "SELL"
