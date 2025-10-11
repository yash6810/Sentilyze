
import os
import pandas as pd
from newsapi import NewsApiClient
import yfinance as yf
from src.utils import get_logger
import time

logger = get_logger(__name__)

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

from datasets import load_dataset

def get_historical_news(ticker):
    """
    Load historical news data for a given ticker from the FNSPID dataset.
    """
    logger.info(f"Loading historical news for {ticker} from Hugging Face...")
    # Load the dataset
    fnspid_dataset = load_dataset("ZIHAN1004/FNSPID")
    # Convert to pandas DataFrame
    df = fnspid_dataset['train'].to_pandas()
    # Filter for the specific ticker
    df_ticker = df[df['ticker'] == ticker]
    return df_ticker

def get_recent_news(ticker, api_key):
    """
    Fetch recent news for a given ticker from the NewsAPI.
    """
    logger.info(f"Fetching recent news for {ticker} from NewsAPI...")
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=ticker,
                                          language='en',
                                          sort_by='publishedAt',
                                          page_size=100)
    articles_df = pd.DataFrame(all_articles['articles'])
    return articles_df

def get_price_history(ticker, period="1y", retries=3, backoff_factor=1):
    cache_path = os.path.join(DATA_DIR, f'{ticker}_price_history.csv')
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(cache_path):
        logger.info(f"Loading price history for {ticker} from cache...")
        history = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
        history.index = pd.to_datetime(history.index, utc=True)
    else:
        logger.info(f"Fetching price history for {ticker} from API...")
        stock = yf.Ticker(ticker)
        for attempt in range(retries):
            try:
                history = stock.history(period=period)
                if history.empty:
                    logger.warning(f"No history data returned for {ticker}. Retrying...")
                    raise ValueError("No data returned")
                history.to_csv(cache_path)
                logger.info(f"Saved price history to {cache_path}")
                return history
            except Exception as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"API call failed for {ticker} with error: {e}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"API call failed for {ticker} after {retries} attempts.")
                    raise e
    return history
