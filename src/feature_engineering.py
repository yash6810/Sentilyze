import pandas as pd


def create_technical_indicators(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators from price history.

    Args:
        price_history (pd.DataFrame): A DataFrame containing historical price data.

    Returns:
        pd.DataFrame: The input DataFrame with added technical indicator columns.
    """
    # 7-day and 21-day moving averages
    price_history["ma7"] = price_history["Close"].rolling(window=7).mean()
    price_history["ma21"] = price_history["Close"].rolling(window=21).mean()

    # Relative Strength Index (RSI)
    delta = price_history["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    price_history["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = price_history["Close"].ewm(span=12, adjust=False).mean()
    exp26 = price_history["Close"].ewm(span=26, adjust=False).mean()
    price_history["macd"] = exp12 - exp26

    # Bollinger Bands
    price_history["bollinger_upper"] = (
        price_history["ma21"] + 2 * price_history["Close"].rolling(window=21).std()
    )
    price_history["bollinger_lower"] = (
        price_history["ma21"] - 2 * price_history["Close"].rolling(window=21).std()
    )

    # Stochastic Oscillator
    low14 = price_history["Low"].rolling(window=14).min()
    high14 = price_history["High"].rolling(window=14).max()
    price_history["stochastic_oscillator"] = 100 * (
        (price_history["Close"] - low14) / (high14 - low14)
    )

    return price_history


def aggregate_sentiment_scores(news_with_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores per day by resampling.

    Args:
        news_with_sentiment (pd.DataFrame): A DataFrame containing news data with a DatetimeIndex.

    Returns:
        pd.DataFrame: A DataFrame with aggregated daily sentiment scores.
    """
    if news_with_sentiment.empty:
        return pd.DataFrame(
            columns=["mean_sentiment_score", "positive", "negative", "neutral"]
        )

    # Resample by day and aggregate sentiment scores
    daily_sentiment = news_with_sentiment.resample("D").agg(
        mean_sentiment_score=("sentiment_score", "mean"),
    )

    # Count sentiment labels per day
    sentiment_counts = pd.get_dummies(news_with_sentiment['sentiment_label']).resample("D").sum()
    daily_sentiment = pd.concat([daily_sentiment, sentiment_counts], axis=1)

    # Ensure all expected sentiment columns exist
    for col in ["positive", "negative", "neutral"]:
        if col not in daily_sentiment.columns:
            daily_sentiment[col] = 0

    # Fill NaN values that result from resampling empty days
    daily_sentiment = daily_sentiment.fillna(0)

    return daily_sentiment


def create_features(
    price_history_with_indicators: pd.DataFrame, daily_sentiment: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges price history with daily sentiment scores to create a feature set.

    Args:
        price_history_with_indicators (pd.DataFrame): A DataFrame containing price history with technical indicators.
        daily_sentiment (pd.DataFrame): A DataFrame containing aggregated daily sentiment scores.

    Returns:
        pd.DataFrame: A merged DataFrame containing the complete feature set.
    """
    merged_df = pd.merge(
        price_history_with_indicators,
        daily_sentiment,
        left_index=True,
        right_index=True,
        how="left",
    )
    merged_df.ffill(inplace=True)
    merged_df.fillna(0, inplace=True)

    # Create the target variable: 1 if next day's close is higher, 0 otherwise
    merged_df["target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(
        int
    )

    return merged_df