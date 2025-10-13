import os
import pandas as pd
from typing import Any
from src.utils import get_logger

os.environ["TRANSFORMERS_BACKEND"] = "pytorch"
logger = get_logger(__name__)

# Define the path for processed (cached) data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def get_sentiment(articles: pd.DataFrame, sentiment_analyzer: Any, ticker: str) -> pd.DataFrame:
    """
    Analyzes the sentiment of news articles, with file-based caching.
    """
    cache_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_sentiment.csv")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        logger.info(f"Loading cached sentiment data for {ticker} from {cache_path}")
        # Load from cache and ensure index is correct
        sentiment_df = pd.read_csv(cache_path, index_col='publishedAt', parse_dates=True)
        return sentiment_df
    else:
        logger.info(f"No cached sentiment data found for {ticker}. Running analysis.")
        articles = articles.dropna(subset=["title", "description"]).copy()
        articles["text"] = articles["title"] + ". " + articles["description"]

        sentiments = sentiment_analyzer(list(articles["text"]))
        articles[["sentiment_label", "sentiment_score"]] = [[s["label"], s["score"]] for s in sentiments]
        
        # Save to cache WITH the index
        articles.to_csv(cache_path, index=True)
        logger.info(f"Saved sentiment data to {cache_path}")
        return articles