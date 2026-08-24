import pytest
from unittest.mock import MagicMock
import pandas as pd
from src.data_sync import sync_all_market_data


def test_sync_all_market_data(mocker):
    """
    Test that sync_all_market_data executes and returns summary metadata.
    """
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="NVDA\nAAPL\n"),
    )

    mock_df = pd.DataFrame(
        {"Close": [100.0, 105.0]}, index=pd.to_datetime(["2026-08-20", "2026-08-21"])
    )
    mock_news = pd.DataFrame({"Title": ["News item 1"]})

    mocker.patch("src.data_sync.get_vix_data", return_value=mock_df)
    mocker.patch("src.data_sync.get_price_history", return_value=mock_df)
    mocker.patch("src.data_sync.get_news", return_value=mock_news)

    summary = sync_all_market_data(period="1y")
    assert summary["assets_synced"] == 2
    assert "NVDA" in summary["details"]
    assert "AAPL" in summary["details"]
    assert summary["details"]["NVDA"]["status"] == "SUCCESS"
    assert summary["details"]["NVDA"]["price_bars"] == 2
