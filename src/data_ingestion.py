import os
import pandas as pd
from newsapi import NewsApiClient
import yfinance as yf
from src.utils import get_logger
import time
from typing import Dict

logger = get_logger(__name__)

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Global variable to track the last yfinance API call time
last_yfinance_call_time = 0
YFINANCE_CALL_INTERVAL = 10 # seconds

def get_news(ticker: str, api_key: str, cache_duration_hours: int = 24, retries: int = 3, backoff_factor: float = 1) -> pd.DataFrame:
    """
    Fetch recent news for a given ticker from the NewsAPI, with caching.

    Args:
        ticker (str): The stock ticker to fetch news for.
        api_key (str): The API key for NewsAPI.
        cache_duration_hours (int, optional): The number of hours to keep the cache before it's considered stale. Defaults to 24.

    Returns:
        pd.DataFrame: A DataFrame containing the recent news articles, indexed by 'publishedAt'.
    """
    cache_path = os.path.join(DATA_DIR, f"{ticker}_news.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    use_cache = False
    if os.path.exists(cache_path):
        # Check if the cache is stale
        cache_age_seconds = time.time() - os.path.getmtime(cache_path)
        if cache_age_seconds < cache_duration_hours * 3600:
            use_cache = True
        else:
            logger.info(f"News cache for {ticker} is stale. Re-fetching...")

    if use_cache:
        logger.info(f"Loading news for {ticker} from cache...")
        articles_df = pd.read_csv(cache_path)
    else:
        logger.info(f"Fetching recent news for {ticker} from NewsAPI...")
        newsapi = NewsApiClient(api_key=api_key)
        for attempt in range(retries):
            try:
                all_articles = newsapi.get_everything(
                    q=ticker, language="en", sort_by="publishedAt", page_size=100
                )
                articles_df = pd.DataFrame(all_articles["articles"])
                break # Break if successful
            except Exception as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"NewsAPI call failed for {ticker} with error: {e}. Retrying in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"NewsAPI call failed for {ticker} after {retries} attempts."
                    )
                    raise e

        articles_df.to_csv(cache_path, index=False)
        logger.info(f"Saved news to {cache_path}")

    # Standardize the DataFrame to have a timezone-aware DatetimeIndex
    articles_df["publishedAt"] = pd.to_datetime(articles_df["publishedAt"], utc=True)
    articles_df = articles_df.set_index("publishedAt").sort_index()

    return articles_df


def get_price_history(ticker: str, period: str = "1y", cache_duration_hours: int = 24, retries: int = 15, backoff_factor: float = 15) -> pd.DataFrame:
    """
    Fetches historical price data for a given ticker from yfinance, with caching.

    Args:
        ticker (str): The stock ticker to fetch price history for.
        period (str, optional): The period for which to fetch the data. Defaults to "1y".
        cache_duration_hours (int, optional): The number of hours to keep the cache before it's considered stale. Defaults to 24.
        retries (int, optional): The number of retries for the API call. Defaults to 3.
        backoff_factor (float, optional): The backoff factor for exponential backoff between retries. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame containing the historical price data.
    """
    global last_yfinance_call_time
    current_time = time.time()
    elapsed_time = current_time - last_yfinance_call_time
    if elapsed_time < YFINANCE_CALL_INTERVAL:
        sleep_duration = YFINANCE_CALL_INTERVAL - elapsed_time
        logger.info(f"Rate limiting yfinance call. Sleeping for {sleep_duration:.2f} seconds.")
        time.sleep(sleep_duration)
    
    last_yfinance_call_time = time.time()
    cache_path = os.path.join(DATA_DIR, f"{ticker}_price_history.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    use_cache = False
    if os.path.exists(cache_path):
        # Check if the cache is stale
        cache_age_seconds = time.time() - os.path.getmtime(cache_path)
        if cache_age_seconds < cache_duration_hours * 3600:
            use_cache = True
        else:
            logger.info(f"Price history cache for {ticker} is stale. Re-fetching...")

    if use_cache:
        logger.info(f"Loading price history for {ticker} from cache...")
        history = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        history.index = pd.to_datetime(history.index, utc=True)
    else:
        logger.info(f"Fetching price history for {ticker} from API...")
        stock = yf.Ticker(ticker)
        for attempt in range(retries):
            try:
                history = stock.history(period=period)
                if history.empty:
                    logger.warning(
                        f"No history data returned for {ticker}. Returning empty DataFrame."
                    )
                    return pd.DataFrame()
                history.to_csv(cache_path)
                logger.info(f"Saved price history to {cache_path}")
                break  # Break out of the loop on success
            except Exception as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2**attempt)
                    logger.warning(
                        f"API call failed for {ticker} with error: {e}. Retrying in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"API call failed for {ticker} after {retries} attempts."
                    )
                    raise e
        # This part is reached if the loop completes (i.e., all retries fail)
        else:
            logger.error(f"Could not fetch price history for {ticker} after {retries} retries.")
            return pd.DataFrame()

    return history