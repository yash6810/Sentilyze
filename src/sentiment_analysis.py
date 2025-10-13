import os
import pandas as pd
from typing import Any
from src.utils import get_logger

os.environ["TRANSFORMERS_BACKEND"] = "pytorch"
logger = get_logger(__name__)

# Define the path for processed (cached) data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def get_sentiment_with_caching(articles: pd.DataFrame, sentiment_analyzer: Any, ticker: str) -> pd.DataFrame:
    """
    Analyzes the sentiment of news articles, with file-based caching for the results.

    Args:
        articles (pd.DataFrame): DataFrame containing news articles.
        sentiment_analyzer (Any): A sentiment analysis pipeline object.
        ticker (str): The stock ticker, used for naming the cache file.

    Returns:
        pd.DataFrame: DataFrame with added sentiment analysis columns.
    """
    cache_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_sentiment.csv")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        logger.info(f"Loading cached sentiment data for {ticker} from {cache_path}")
        return pd.read_csv(cache_path)
    else:
        logger.info(f"No cached sentiment data found for {ticker}. Running analysis.")
        sentiment_df = get_sentiment(articles, sentiment_analyzer)
        sentiment_df.to_csv(cache_path, index=False)
        logger.info(f"Saved sentiment data to {cache_path}")
        return sentiment_df

def get_sentiment(articles: pd.DataFrame, sentiment_analyzer: Any) -> pd.DataFrame:
    """
    Analyzes the sentiment of news articles.

    Args:
        articles (pd.DataFrame): A DataFrame containing news articles with 'title' and 'description' columns.
        sentiment_analyzer (Any): A sentiment analysis pipeline object.

    Returns:
        pd.DataFrame: The input DataFrame with added 'sentiment_label' and 'sentiment_score' columns.
    """
    articles = articles.dropna(subset=["title", "description"]).copy()
    articles["text"] = articles["title"] + ". " + articles["description"]

    # Process all articles
    sentiments = sentiment_analyzer(list(articles["text"]))

    articles[["sentiment_label", "sentiment_score"]] = [
        [s["label"], s["score"]] for s in sentiments
    ]

    return articles
