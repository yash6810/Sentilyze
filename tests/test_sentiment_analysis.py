import pytest
from unittest.mock import MagicMock
import pandas as pd
import os
import time
from src.sentiment_analysis import get_sentiment, PROCESSED_DATA_DIR

@pytest.fixture(autouse=True)
def temp_data_dir(tmpdir):
    """Fixture to set a temporary data directory for tests."""
    original_data_dir = PROCESSED_DATA_DIR
    new_data_dir = tmpdir.mkdir("test_processed_data")
    # Ugly but necessary: we need to modify the global variable in the module
    # under test.
    import src.sentiment_analysis
    src.sentiment_analysis.PROCESSED_DATA_DIR = str(new_data_dir)
    yield str(new_data_dir)
    src.sentiment_analysis.PROCESSED_DATA_DIR = original_data_dir

def test_get_sentiment_analyzes_and_enriches_data(temp_data_dir):
    """
    Test that get_sentiment correctly analyzes sentiment and adds the right columns.
    """
    # Arrange
    articles = pd.DataFrame([
        {'Title': 'Positive news', 'description': 'Things are great!', 'Date': '2023-01-01'},
        {'Title': 'Negative news', 'description': 'Things are bad.', 'Date': '2023-01-02'}
    ])
    articles['Date'] = pd.to_datetime(articles['Date'])
    
    mock_sentiment_analyzer = MagicMock()
    mock_sentiment_analyzer.return_value = [
        {'label': 'Positive', 'score': 0.9},
        {'label': 'Negative', 'score': 0.8}
    ]
    
    # Act
    sentiment_df = get_sentiment(articles, mock_sentiment_analyzer, ticker="TEST")
    
    # Assert
    assert 'sentiment_label' in sentiment_df.columns
    assert 'sentiment_score' in sentiment_df.columns
    assert sentiment_df.loc[0, 'sentiment_label'] == 'Positive'
    assert sentiment_df.loc[0, 'sentiment_score'] == 0.9
    assert sentiment_df.loc[1, 'sentiment_label'] == 'Negative'
    assert sentiment_df.loc[1, 'sentiment_score'] == 0.8


def test_get_sentiment_loads_from_cache(temp_data_dir):
    """
    Test that get_sentiment loads data from the cache if it's not stale.
    """
    # Arrange
    ticker = "TEST"
    cache_path = os.path.join(temp_data_dir, f"{ticker}_sentiment.csv")
    mock_cached_data = pd.DataFrame({
        'Title': ['Cached Title'], 
        'publishedAt': ['2023-01-02T12:00:00Z'],
        'sentiment_label': ['Positive'],
        'sentiment_score': [0.99]
    })
    mock_cached_data.to_csv(cache_path, index=False)
    
    mock_sentiment_analyzer = MagicMock()

    # Act
    sentiment_df = get_sentiment(pd.DataFrame(), mock_sentiment_analyzer, ticker=ticker)

    # Assert
    assert sentiment_df.iloc[0]['Title'] == 'Cached Title'
    mock_sentiment_analyzer.assert_not_called()

def test_get_sentiment_handles_missing_text_columns():
    """
    Test that get_sentiment handles DataFrames with missing text columns gracefully.
    """
    # Arrange
    articles = pd.DataFrame([{'Date': '2023-01-01'}])
    articles['Date'] = pd.to_datetime(articles['Date'])
    mock_sentiment_analyzer = MagicMock()

    # Act
    result_df = get_sentiment(articles, mock_sentiment_analyzer, ticker="TEST")

    # Assert
    assert "sentiment_label" not in result_df.columns
    mock_sentiment_analyzer.assert_not_called()

def test_get_sentiment_bypasses_cache_if_no_ticker():
    """
    Test that get_sentiment bypasses the cache when no ticker is provided.
    """
    # Arrange
    articles = pd.DataFrame([
        {'Title': 'Positive news', 'description': 'Things are great!', 'Date': '2023-01-01'}
    ])
    articles['Date'] = pd.to_datetime(articles['Date'])
    
    mock_sentiment_analyzer = MagicMock(return_value=[{'label': 'Positive', 'score': 0.9}])

    # Act
    sentiment_df = get_sentiment(articles, mock_sentiment_analyzer, ticker=None)

    # Assert
    assert 'sentiment_label' in sentiment_df.columns
    assert 'sentiment_score' in sentiment_df.columns
    mock_sentiment_analyzer.assert_called_once()