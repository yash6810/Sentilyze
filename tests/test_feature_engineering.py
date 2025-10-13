import pandas as pd
import pytest
from src.feature_engineering import create_technical_indicators, aggregate_sentiment_scores, create_features

@pytest.fixture
def sample_price_history() -> pd.DataFrame:
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'Close': [100, 102, 101, 103, 105],
        'Low': [99, 101, 100, 102, 104],
        'High': [101, 103, 102, 104, 106]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

@pytest.fixture
def sample_news_with_sentiment() -> pd.DataFrame:
    data = {
        'publishedAt': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'sentiment_score': [0.8, 0.2, -0.5],
        'sentiment_label': ['positive', 'neutral', 'negative']
    }
    df = pd.DataFrame(data)
    df.set_index('publishedAt', inplace=True)
    return df

def test_create_technical_indicators(sample_price_history):
    df = create_technical_indicators(sample_price_history)
    assert 'ma7' in df.columns
    assert 'ma21' in df.columns
    assert 'rsi' in df.columns
    assert 'macd' in df.columns
    assert 'bollinger_upper' in df.columns
    assert 'bollinger_lower' in df.columns
    assert 'stochastic_oscillator' in df.columns

def test_aggregate_sentiment_scores(sample_news_with_sentiment):
    daily_sentiment = aggregate_sentiment_scores(sample_news_with_sentiment)
    assert 'mean_sentiment_score' in daily_sentiment.columns
    assert 'positive' in daily_sentiment.columns
    assert 'negative' in daily_sentiment.columns
    assert 'neutral' in daily_sentiment.columns
    # Check values on the correct date index
    assert daily_sentiment.loc['2023-01-01']['positive'] == 1
    assert daily_sentiment.loc['2023-01-01']['neutral'] == 1
    assert daily_sentiment.loc['2023-01-02']['negative'] == 1

def test_create_features(sample_price_history, sample_news_with_sentiment):
    price_history_with_indicators = create_technical_indicators(sample_price_history)
    daily_sentiment = aggregate_sentiment_scores(sample_news_with_sentiment)
    features_df = create_features(price_history_with_indicators, daily_sentiment)
    assert 'mean_sentiment_score' in features_df.columns
    assert not features_df.isnull().values.any()