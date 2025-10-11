import pandas as pd

def create_technical_indicators(price_history):
    """
    Create technical indicators from price history.
    """
    # 7-day and 21-day moving averages
    price_history['ma7'] = price_history['Close'].rolling(window=7).mean()
    price_history['ma21'] = price_history['Close'].rolling(window=21).mean()

    # Relative Strength Index (RSI)
    delta = price_history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    price_history['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = price_history['Close'].ewm(span=12, adjust=False).mean()
    exp26 = price_history['Close'].ewm(span=26, adjust=False).mean()
    price_history['macd'] = exp12 - exp26

    # Bollinger Bands
    price_history['bollinger_upper'] = price_history['ma21'] + 2 * price_history['Close'].rolling(window=21).std()
    price_history['bollinger_lower'] = price_history['ma21'] - 2 * price_history['Close'].rolling(window=21).std()

    # Stochastic Oscillator
    low14 = price_history['Low'].rolling(window=14).min()
    high14 = price_history['High'].rolling(window=14).max()
    price_history['stochastic_oscillator'] = 100 * ((price_history['Close'] - low14) / (high14 - low14))

    return price_history

def aggregate_sentiment_scores(news_with_sentiment):
    """
    Aggregate sentiment scores per day.
    """
    if news_with_sentiment.empty:
        return pd.DataFrame(columns=['date', 'mean_sentiment_score', 'positive', 'negative', 'neutral'])


    news_with_sentiment['date'] = news_with_sentiment['publishedAt'].dt.date

    daily_sentiment = news_with_sentiment.groupby('date').agg(
        mean_sentiment_score=('sentiment_score', 'mean'),
    )
    sentiment_counts = news_with_sentiment.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
    daily_sentiment = pd.concat([daily_sentiment, sentiment_counts], axis=1)

    # Ensure sentiment columns exist
    for col in ['positive', 'negative', 'neutral']:
        if col not in daily_sentiment.columns:
            daily_sentiment[col] = 0

    # Fill NaN values
    daily_sentiment = daily_sentiment.fillna(0)

    return daily_sentiment

def create_features(price_history_with_indicators, daily_sentiment):
    price_history_with_indicators['date'] = price_history_with_indicators.index.date
    merged_df = pd.merge(price_history_with_indicators, daily_sentiment, on='date', how='left')
    merged_df.ffill(inplace=True)
    merged_df.fillna(0, inplace=True)
    return merged_df