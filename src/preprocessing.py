import pandas as pd
import yfinance as yf
import os
import time
from src.utils import get_logger
from src.data_ingestion import get_price_history, get_news, get_vix_data
from src.sentiment_analysis import get_sentiment
from src.feature_engineering import (
    create_technical_indicators,
    aggregate_sentiment_scores,
    create_features,
)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
from typing import Any
import concurrent.futures

logger = get_logger(__name__)


# Cache the sentiment analyzer to avoid reloading on every call
@lru_cache(maxsize=1)
def _load_sentiment_analyzer() -> Any:
    """
    Loads the FinBERT sentiment analysis model and tokenizer from the local
    './models/finbert-fine-tuned' directory and returns a Hugging Face pipeline.
    """
    logger.info("Loading FinBERT sentiment analysis model for preprocessing...")
    tokenizer = AutoTokenizer.from_pretrained("./models/finbert-fine-tuned")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./models/finbert-fine-tuned"
    )
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def clean_headline_data(
    input_path: str, output_path: str, cache_dir: str = "data/processed"
):
    """
    Cleans a headline CSV file by removing rows with invalid stock tickers.
    Caches the list of valid and invalid tickers to speed up subsequent runs.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to save the cleaned CSV file.
        cache_dir (str): The directory to store ticker validation cache files.
    """
    logger.info(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path, encoding="ISO-8859-1")

    if "headline" in df.columns:
        df.rename(
            columns={"date": "Date", "headline": "Title", "stock": "Ticker"},
            inplace=True,
        )

    original_rows = len(df)
    unique_tickers = df["Ticker"].unique()

    # --- Ticker Validation Caching ---
    os.makedirs(cache_dir, exist_ok=True)
    valid_tickers_cache_path = os.path.join(cache_dir, "valid_tickers.json")

    if os.path.exists(valid_tickers_cache_path):
        logger.info(f"Loading valid tickers from cache: {valid_tickers_cache_path}")
        with open(valid_tickers_cache_path, "r") as f:
            import json

            valid_tickers = json.load(f)
    else:
        from tqdm import tqdm

        logger.info("No valid tickers cache found. Validating all tickers...")
        valid_tickers = []
        invalid_tickers = []

        for ticker in tqdm(unique_tickers, desc="Validating tickers"):
            is_valid = False
            for attempt in range(3):
                try:
                    stock = yf.Ticker(ticker)
                    if not stock.history(period="1d").empty:
                        is_valid = True
                        break
                except Exception as e:
                    logger.warning(
                        f"Error validating {ticker} on attempt {attempt + 1}: {e}"
                    )
                    time.sleep(2)  # Wait before retrying

            if is_valid:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)

        logger.info(
            f"Found {len(valid_tickers)} valid tickers and {len(invalid_tickers)} invalid tickers."
        )
        if invalid_tickers:
            logger.info(f"Invalid tickers found: {invalid_tickers[:20]}...")

        # Cache the results
        with open(valid_tickers_cache_path, "w") as f:
            import json

            json.dump(valid_tickers, f)
        logger.info(f"Saved valid tickers to cache: {valid_tickers_cache_path}")

    # Filter the DataFrame
    cleaned_df = df[df["Ticker"].isin(valid_tickers)]
    cleaned_rows = len(cleaned_df)

    logger.info(f"Removed {original_rows - cleaned_rows} rows with invalid tickers.")

    logger.info(f"Saving cleaned data to {output_path}...")
    cleaned_df.to_csv(output_path, index=False)
    logger.info("Done.")


def _get_api_key() -> str | None:
    """Safely attempts to retrieve the API key from Streamlit secrets, falling back to environment variables."""
    try:
        import streamlit as st

        try:
            if "NEWS_API_KEY" in st.secrets:
                return st.secrets["NEWS_API_KEY"]
        except Exception:
            pass
    except ImportError:
        pass
    return os.environ.get("NEWS_API_KEY")


def preprocess_data(
    ticker: str, period: str = "10y", use_cache: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the data acquisition, sentiment analysis, and feature engineering
    for a given ticker.

    Args:
        ticker (str): The stock ticker to preprocess data for.
        period (str): The time period for the data (e.g., "10y", "5y", "max").
        use_cache (bool): If True, aggressively favors cached data (1 year duration).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - features_df: A DataFrame with engineered features ready for model training.
            - price_history_with_indicators: The price history with technical indicators.
            - news_with_sentiment_df: The news headlines with sentiment scores.
    """
    logger.info(f"Starting data preprocessing for {ticker} (Use Cache: {use_cache})...")

    # 1. Fetch data sequentially to avoid yfinance rate limits
    logger.info("Fetching data...")
    api_key = _get_api_key()

    # Use 1 year cache duration if use_cache is True
    cache_duration = 8760 if use_cache else 24

    news_df = get_news(ticker, api_key, cache_duration_hours=cache_duration)
    price_history_df = get_price_history(
        ticker, period, cache_duration_hours=cache_duration
    )
    vix_df = get_vix_data(period, cache_duration_hours=cache_duration)

    # 2. Analyze sentiment
    logger.info("Analyzing sentiment...")
    sentiment_analyzer = _load_sentiment_analyzer()
    news_with_sentiment_df = get_sentiment(
        news_df, sentiment_analyzer, ticker, cache_duration_hours=cache_duration
    )

    # 3. Feature Engineering
    logger.info("Creating features...")
    price_history_with_indicators = create_technical_indicators(price_history_df)
    daily_sentiment = aggregate_sentiment_scores(news_with_sentiment_df)
    features_df = create_features(
        price_history_with_indicators, daily_sentiment, vix_df
    )
    features_df = features_df.dropna().sort_index()

    logger.info(f"Preprocessing for {ticker} complete. Shape: {features_df.shape}")
    return features_df, price_history_with_indicators, news_with_sentiment_df
