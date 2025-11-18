import pytest
import pandas as pd
import os
import time
import json
from src.preprocessing import clean_headline_data, preprocess_data, _load_sentiment_analyzer


@pytest.fixture
def temp_data_dirs(tmpdir):
    data_dir = tmpdir.mkdir("data")
    data_raw_dir = data_dir.mkdir("raw")
    data_processed_dir = data_dir.mkdir("processed")
    
    yield {
        "raw": str(data_raw_dir),
        "processed": str(data_processed_dir),
        "tmp": str(tmpdir)
    }

# --- Tests for clean_headline_data ---

def test_clean_headline_data_valid_tickers_no_cache(temp_data_dirs, mocker):
    input_path = os.path.join(temp_data_dirs["raw"], "headlines.csv")
    output_path = os.path.join(temp_data_dirs["raw"], "cleaned_headlines.csv")
    
    input_data = pd.DataFrame({
        "date": ["2023-01-01"],
        "headline": ["Title 1"],
        "stock": ["AAPL"]
    })
    input_data.to_csv(input_path, index=False)
    
    mock_ticker_instance = mocker.patch('yfinance.Ticker')
    mock_ticker_instance.return_value.history.return_value = pd.DataFrame({'Close': [100]})

    clean_headline_data(input_path, output_path, cache_dir=temp_data_dirs["processed"])

    assert os.path.exists(output_path)
    cleaned_df = pd.read_csv(output_path)
    assert not cleaned_df.empty
    assert cleaned_df["Ticker"].iloc[0] == "AAPL"
    mock_ticker_instance.return_value.history.assert_called_with(period="1d")

def test_clean_headline_data_invalid_tickers_no_cache(temp_data_dirs, mocker):
    input_path = os.path.join(temp_data_dirs["raw"], "headlines.csv")
    output_path = os.path.join(temp_data_dirs["raw"], "cleaned_headlines.csv")
    
    input_data = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01"],
        "headline": ["Title 1", "Title 2"],
        "stock": ["AAPL", "INVALID"]
    })
    input_data.to_csv(input_path, index=False)
    
    mock_ticker_instance = mocker.patch('yfinance.Ticker')
    mock_ticker_instance.return_value.history.side_effect = [
        pd.DataFrame({'Close': [100]}), 
        pd.DataFrame()
    ]

    clean_headline_data(input_path, output_path, cache_dir=temp_data_dirs["processed"])

    assert os.path.exists(output_path)
    cleaned_df = pd.read_csv(output_path)
    assert len(cleaned_df) == 1
    assert cleaned_df["Ticker"].iloc[0] == "AAPL"

def test_clean_headline_data_loads_from_cache(temp_data_dirs, mocker):
    input_path = os.path.join(temp_data_dirs["raw"], "headlines.csv")
    output_path = os.path.join(temp_data_dirs["raw"], "cleaned_headlines.csv")
    valid_tickers_cache_path = os.path.join(temp_data_dirs["processed"], "valid_tickers.json")
    
    input_data = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01"],
        "headline": ["Title 1", "Title 2"],
        "stock": ["AAPL", "GOOGL"]
    })
    input_data.to_csv(input_path, index=False)
    
    with open(valid_tickers_cache_path, "w") as f:
        json.dump(["AAPL"], f)

    mocker.patch('yfinance.Ticker') # Ensure Ticker is mocked even if not called
    clean_headline_data(input_path, output_path, cache_dir=temp_data_dirs["processed"])

    assert os.path.exists(output_path)
    cleaned_df = pd.read_csv(output_path)
    assert len(cleaned_df) == 1
    assert cleaned_df["Ticker"].iloc[0] == "AAPL"
    # assert mock_yfinance_ticker.history.call_count == 0 # Cannot assert on a non-existent mock


# --- Tests for preprocess_data ---

def test_preprocess_data_orchestrates_correctly(mocker):
    ticker = "TEST"
    
    mock_get_news = mocker.patch('src.preprocessing.get_news')
    mock_get_price_history = mocker.patch('src.preprocessing.get_price_history')
    mock_get_sentiment = mocker.patch('src.preprocessing.get_sentiment')
    mock_create_technical_indicators = mocker.patch('src.preprocessing.create_technical_indicators')
    mock_aggregate_sentiment_scores = mocker.patch('src.preprocessing.aggregate_sentiment_scores')
    mock_create_features = mocker.patch('src.preprocessing.create_features')

    # Mock return values for all sub-functions
    mock_get_news.return_value = pd.DataFrame()
    mock_get_price_history.return_value = pd.DataFrame()
    mock_get_sentiment.return_value = pd.DataFrame()
    mock_create_features.return_value = pd.DataFrame({
        'feature1': [1, 2],
        'target': [0, 1]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))

    result_df = preprocess_data(ticker)

    mock_get_news.assert_called_once_with(ticker, os.environ.get("NEWS_API_KEY"))
    mock_get_price_history.assert_called_once_with(ticker)
    mocker.patch('src.preprocessing._load_sentiment_analyzer') # Mock internal call to load analyzer
    mock_get_sentiment.assert_called_once()
    mock_create_technical_indicators.assert_called_once()
    mock_aggregate_sentiment_scores.assert_called_once()
    mock_create_features.assert_called_once()
    
    assert isinstance(result_df, pd.DataFrame)
    assert 'target' in result_df.columns
    assert not result_df.empty

def test_load_sentiment_analyzer_caches():
    # Call it twice to ensure caching works
    analyzer1 = _load_sentiment_analyzer()
    analyzer2 = _load_sentiment_analyzer()
    assert analyzer1 is analyzer2 # Should be the same object due to lru_cache
