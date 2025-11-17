import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import time
from src.data_ingestion import get_news, get_price_history, DATA_DIR

@pytest.fixture(autouse=True)
def temp_data_dir(tmpdir):
    """Fixture to set a temporary data directory for tests."""
    original_data_dir = DATA_DIR
    new_data_dir = tmpdir.mkdir("test_data")
    # Ugly but necessary: we need to modify the global variable in the module
    # under test.
    import src.data_ingestion
    src.data_ingestion.DATA_DIR = str(new_data_dir)
    yield str(new_data_dir)
    src.data_ingestion.DATA_DIR = original_data_dir


def test_get_price_history_fetches_and_caches_data(mocker, temp_data_dir):
    """
    Test that get_price_history fetches data from yfinance and saves it to a cache file.
    """
    # Arrange
    ticker = "TEST"
    mock_history = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    mock_history.index.name = 'Date'
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_history
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    # Act
    history = get_price_history(ticker, period="1y")

    # Assert
    assert not history.empty
    pd.testing.assert_frame_equal(history, mock_history)
    cache_path = os.path.join(temp_data_dir, f"{ticker}_price_history.csv")
    assert os.path.exists(cache_path)


def test_get_price_history_loads_from_cache(mocker, temp_data_dir):
    """
    Test that get_price_history loads data from the cache if it's not stale.
    """
    # Arrange
    ticker = "TEST"
    cache_path = os.path.join(temp_data_dir, f"{ticker}_price_history.csv")
    mock_cached_data = pd.DataFrame({'Close': [90, 91, 92]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    mock_cached_data.index.name = 'Date'
    mock_cached_data.to_csv(cache_path)
    
    mock_ticker = MagicMock()
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    # Act
    history = get_price_history(ticker, period="1y")

    # Assert
    # We need to parse the dates in the index when reading from csv
    expected_df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
    expected_df.index = pd.to_datetime(expected_df.index, utc=True)
    pd.testing.assert_frame_equal(history, expected_df)
    mock_ticker.history.assert_not_called()


def test_get_price_history_refetches_stale_cache(mocker, temp_data_dir):
    """
    Test that get_price_history re-fetches data if the cache is stale.
    """
    # Arrange
    ticker = "TEST"
    cache_path = os.path.join(temp_data_dir, f"{ticker}_price_history.csv")
    mock_cached_data = pd.DataFrame({'Close': [90, 91, 92]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    mock_cached_data.index.name = 'Date'
    mock_cached_data.to_csv(cache_path)
    
    # Make the cache file seem old
    one_day_ago = time.time() - 25 * 3600
    os.utime(cache_path, (one_day_ago, one_day_ago))

    mock_fresh_data = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.to_datetime(['2023-01-04', '2023-01-05', '2023-01-06']))
    mock_fresh_data.index.name = 'Date'
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_fresh_data
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    # Act
    history = get_price_history(ticker, period="1y", cache_duration_hours=24)

    # Assert
    pd.testing.assert_frame_equal(history, mock_fresh_data)
    mock_ticker.history.assert_called_once()


def test_get_news_fetches_and_caches_data(mocker, temp_data_dir):
    """
    Test that get_news fetches data from NewsAPI and saves it to a cache file.
    """
    # Arrange
    ticker = "TEST"
    api_key = "test_api_key"
    mock_articles = {'articles': [{'title': 'Test Title', 'description': 'Test Description', 'publishedAt': '2023-01-01T12:00:00Z'}]}
    
    mock_newsapi_client = MagicMock()
    mock_newsapi_client.get_everything.return_value = mock_articles
    mocker.patch('src.data_ingestion.NewsApiClient', return_value=mock_newsapi_client)

    # Act
    news_df = get_news(ticker, api_key)

    # Assert
    assert not news_df.empty
    assert news_df.iloc[0]['title'] == 'Test Title'
    cache_path = os.path.join(temp_data_dir, f"{ticker}_news.csv")
    assert os.path.exists(cache_path)


def test_get_news_loads_from_cache(mocker, temp_data_dir):
    """
    Test that get_news loads data from the cache if it's not stale.
    """
    # Arrange
    ticker = "TEST"
    api_key = "test_api_key"
    cache_path = os.path.join(temp_data_dir, f"{ticker}_news.csv")
    mock_cached_data = pd.DataFrame({'title': ['Cached Title'], 'description': ['Cached Desc'], 'publishedAt': ['2023-01-02T12:00:00Z']})
    mock_cached_data.to_csv(cache_path, index=False)

    mock_newsapi_client = MagicMock()
    mocker.patch('src.data_ingestion.NewsApiClient', return_value=mock_newsapi_client)

    # Act
    news_df = get_news(ticker, api_key)

    # Assert
    assert news_df.iloc[0]['title'] == 'Cached Title'
    mock_newsapi_client.get_everything.assert_not_called()


def test_get_news_refetches_stale_cache(mocker, temp_data_dir):
    """
    Test that get_news re-fetches data if the cache is stale.
    """
    # Arrange
    ticker = "TEST"
    api_key = "test_api_key"
    cache_path = os.path.join(temp_data_dir, f"{ticker}_news.csv")
    mock_cached_data = pd.DataFrame({'title': ['Cached Title'], 'publishedAt': ['2023-01-02T12:00:00Z']})
    mock_cached_data.to_csv(cache_path, index=False)

    # Make the cache file seem old
    two_days_ago = time.time() - 49 * 3600
    os.utime(cache_path, (two_days_ago, two_days_ago))

    mock_fresh_articles = {'articles': [{'title': 'Fresh Title', 'description': 'Fresh Desc', 'publishedAt': '2023-01-03T12:00:00Z'}]}
    mock_newsapi_client = MagicMock()
    mock_newsapi_client.get_everything.return_value = mock_fresh_articles
    mocker.patch('src.data_ingestion.NewsApiClient', return_value=mock_newsapi_client)

    # Act
    news_df = get_news(ticker, api_key, cache_duration_hours=48)

    # Assert
    assert news_df.iloc[0]['title'] == 'Fresh Title'
    mock_newsapi_client.get_everything.assert_called_once()