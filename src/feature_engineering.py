import pandas as pd


def create_technical_indicators(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators from price history.

    Args:
        price_history (pd.DataFrame): A DataFrame containing historical price data.

    Returns:
        pd.DataFrame: The input DataFrame with added technical indicator columns.
    """
    # SHIFT price data to prevent lookahead bias.
    ph_shifted = price_history.shift(1)

    # Moving averages including 200-day for regime filter
    price_history["ma7"] = ph_shifted["Close"].rolling(window=7).mean()
    price_history["ma21"] = ph_shifted["Close"].rolling(window=21).mean()
    price_history["sma200"] = ph_shifted["Close"].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = ph_shifted["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)  # Add epsilon to prevent division by zero
    price_history["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = ph_shifted["Close"].ewm(span=12, adjust=False).mean()
    exp26 = ph_shifted["Close"].ewm(span=26, adjust=False).mean()
    price_history["macd"] = exp12 - exp26

    # Bollinger Bands
    price_history["bollinger_upper"] = (
        price_history["ma21"] + 2 * ph_shifted["Close"].rolling(window=21).std()
    )
    price_history["bollinger_lower"] = (
        price_history["ma21"] - 2 * ph_shifted["Close"].rolling(window=21).std()
    )

    # Stochastic Oscillator
    low14 = ph_shifted["Low"].rolling(window=14).min()
    high14 = ph_shifted["High"].rolling(window=14).max()
    price_history["stochastic_oscillator"] = 100 * (
        (ph_shifted["Close"] - low14) / (high14 - low14 + 1e-10)
    )

    # Average True Range (ATR)
    high_low = ph_shifted["High"] - ph_shifted["Low"]
    high_close = (ph_shifted["High"] - price_history["Close"].shift(2)).abs()
    low_close = (ph_shifted["Low"] - price_history["Close"].shift(2)).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    price_history["atr"] = true_range.rolling(14).mean()

    # Short and medium-term price momentum (zero-lookahead since using ph_shifted)
    price_history["return_1d"] = ph_shifted["Close"].pct_change(1)
    price_history["return_5d"] = ph_shifted["Close"].pct_change(5)
    price_history["return_21d"] = ph_shifted["Close"].pct_change(21)
    price_history["volatility_21d"] = (
        ph_shifted["Close"].pct_change(1).rolling(21).std()
    )
    price_history["price_to_sma200"] = ph_shifted["Close"] / (
        price_history["sma200"] + 1e-10
    )
    price_history["rsi_slope"] = price_history["rsi"].diff(3)

    # Enhanced momentum, volume, and volatility ratio features
    price_history["ma_spread"] = (
        (
            (price_history["ma7"] - price_history["ma21"])
            / (price_history["ma21"] + 1e-5)
        )
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
    )

    if "Volume" in ph_shifted.columns:
        price_history["volume_ratio"] = (
            (ph_shifted["Volume"] / (ph_shifted["Volume"].rolling(20).mean() + 1e-5))
            .replace([float("inf"), float("-inf")], 1.0)
            .fillna(1.0)
        )
    else:
        price_history["volume_ratio"] = 1.0

    price_history["atr_ratio"] = (
        (price_history["atr"] / (ph_shifted["Close"] + 1e-5))
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
    )

    # =========================================================================
    # High-Order Formulaic Alpha Factors (Qlib-Inspired Microstructure Signals)
    # =========================================================================
    # 1. 20-Day Rolling Return Skewness (Asymmetry of price innovations)
    ret_1d = ph_shifted["Close"].pct_change(1)
    price_history["alpha_vol_skew_20d"] = (
        ret_1d.rolling(20)
        .skew()
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
    )

    # 2. Intraday Range Expansion Volatility Ratio (vs 50-day baseline)
    atr_50 = true_range.rolling(50).mean() + 1e-5
    price_history["alpha_range_vol_ratio"] = (
        (true_range / atr_50).replace([float("inf"), float("-inf")], 1.0).fillna(1.0)
    )

    # 3. 5-Day Acceleration of Chaikin Money Flow (CMF derivative)
    if "Volume" in ph_shifted.columns:
        cl_spread = (ph_shifted["Close"] - ph_shifted["Low"]) - (
            ph_shifted["High"] - ph_shifted["Close"]
        )
        hl_diff = ph_shifted["High"] - ph_shifted["Low"] + 1e-5
        mf_multiplier = cl_spread / hl_diff
        mf_volume = mf_multiplier * ph_shifted["Volume"]
        cmf_20 = mf_volume.rolling(20).sum() / (
            ph_shifted["Volume"].rolling(20).sum() + 1e-5
        )
        price_history["alpha_money_flow_accel"] = (
            cmf_20.diff(5).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        )
    else:
        price_history["alpha_money_flow_accel"] = 0.0

    # 4. Normalized Trend Residuals (Z-score of price vs 21-day EWMA)
    ewma21 = ph_shifted["Close"].ewm(span=21, adjust=False).mean()
    roll_std21 = ph_shifted["Close"].rolling(21).std() + 1e-5
    price_history["alpha_trend_residual"] = (
        ((ph_shifted["Close"] - ewma21) / roll_std21)
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
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

    # Resample by day and aggregate sentiment scores, then normalize index
    daily_sentiment = news_with_sentiment.resample("D").agg(
        mean_sentiment_score=("sentiment_score", "mean"),
    )
    daily_sentiment.index = daily_sentiment.index.normalize()

    # Convert sentiment labels to lowercase before creating dummies
    news_with_sentiment["sentiment_label"] = news_with_sentiment[
        "sentiment_label"
    ].str.lower()

    # Count sentiment labels per day
    sentiment_counts = (
        pd.get_dummies(news_with_sentiment["sentiment_label"]).resample("D").sum()
    )
    daily_sentiment = pd.concat([daily_sentiment, sentiment_counts], axis=1)

    # Ensure all expected sentiment columns exist
    for col in ["positive", "negative", "neutral"]:
        if col not in daily_sentiment.columns:
            daily_sentiment[col] = 0

    # Fill NaN values that result from resampling empty days
    daily_sentiment = daily_sentiment.fillna(0)

    return daily_sentiment


def create_features(
    price_history_with_indicators: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    vix_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merges price history with daily sentiment scores and VIX data to create a feature set.

    Args:
        price_history_with_indicators (pd.DataFrame): A DataFrame containing price history with technical indicators.
        daily_sentiment (pd.DataFrame): A DataFrame containing aggregated daily sentiment scores.
        vix_data (pd.DataFrame, optional): A DataFrame containing VIX history.

    Returns:
        pd.DataFrame: A merged DataFrame containing the complete feature set.
    """
    # Normalize indices to ensure clean alignment on midnight UTC
    price_history_with_indicators.index = (
        price_history_with_indicators.index.normalize()
    )
    daily_sentiment.index = daily_sentiment.index.normalize()

    # SHIFT sentiment data to prevent lookahead bias (use yesterday's news for today's prediction)
    daily_sentiment = daily_sentiment.shift(1)

    merged_df = pd.merge(
        price_history_with_indicators,
        daily_sentiment,
        left_index=True,
        right_index=True,
        how="left",
    )

    # Merge VIX if provided
    if vix_data is not None and not vix_data.empty:
        # Keep only the Close column from VIX and rename it
        vix_subset = vix_data[["Close"]].rename(columns={"Close": "vix_close"})
        # Calculate 5-day moving average of VIX
        vix_subset["vix_ma5"] = vix_subset["vix_close"].rolling(window=5).mean()
        vix_subset["vix_change_1d"] = vix_subset["vix_close"].pct_change(1)
        vix_subset = vix_subset.shift(1)

        merged_df = pd.merge(
            merged_df, vix_subset, left_index=True, right_index=True, how="left"
        )
    else:
        # If no VIX data, fill with 0 (or some default) just to keep columns present
        merged_df["vix_close"] = 0
        merged_df["vix_ma5"] = 0
        merged_df["vix_change_1d"] = 0

    merged_df.ffill(inplace=True)
    merged_df.replace([float("inf"), float("-inf")], 0.0, inplace=True)
    merged_df.fillna(0.0, inplace=True)

    # Create the target variable: 1 if next day's close is higher, 0 otherwise
    merged_df["target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(
        int
    )

    # Drop raw price data to prevent leakage
    cols_to_drop = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    merged_df = merged_df.drop(columns=cols_to_drop, errors="ignore")

    return merged_df
