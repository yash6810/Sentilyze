import os
import pandas as pd
from typing import Any
from src.utils import get_logger

os.environ["TRANSFORMERS_BACKEND"] = "pytorch"
logger = get_logger(__name__)

# Define the path for processed (cached) data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def get_sentiment(articles: pd.DataFrame, sentiment_analyzer: Any, ticker: str | None = None) -> pd.DataFrame:
    """
    Analyzes the sentiment of news articles, with optional file-based caching.
    If ticker is None, caching is bypassed.
    """
    if ticker:
        cache_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_sentiment.csv")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        if os.path.exists(cache_path):
            logger.info(f"Loading cached sentiment data for {ticker} from {cache_path}")
            # Load from cache and ensure index is correct
            sentiment_df = pd.read_csv(cache_path, index_col='publishedAt', parse_dates=True)
            return sentiment_df
    
    logger.info(f"Running sentiment analysis (caching {'enabled' if ticker else 'disabled'}).")
    
    from tqdm import tqdm

    # Determine available text columns and construct 'text' for sentiment analysis
    text_columns = []
    if "Title" in articles.columns:
        text_columns.append("Title")
    if "description" in articles.columns:
        text_columns.append("description")
    
    if not text_columns:
        logger.warning("No 'Title' or 'description' columns found for sentiment analysis. Skipping.")
        return articles # Return original articles if no text columns

    # Create a temporary DataFrame for sentiment analysis
    articles_for_sentiment = articles[text_columns + ['Date']].dropna(subset=text_columns).copy()

    if "Title" in text_columns and "description" in text_columns:
        articles_for_sentiment["text"] = articles_for_sentiment["Title"] + ". " + articles_for_sentiment["description"]
    elif "Title" in text_columns:
        articles_for_sentiment["text"] = articles_for_sentiment["Title"]
    elif "description" in text_columns:
        articles_for_sentiment["text"] = articles_for_sentiment["description"]

    results = []
    text_list = articles_for_sentiment["text"].tolist()
    chunk_size = 32 # Process 32 articles at a time
    
    for i in tqdm(range(0, len(text_list), chunk_size), desc="Analyzing sentiment"):
        chunk = text_list[i:i+chunk_size]
        results.extend(sentiment_analyzer(chunk))

    articles_for_sentiment[["sentiment_label", "sentiment_score"]] = [[s["label"], s["score"]] for s in results]

    # Merge sentiment back to original articles DataFrame
    articles = pd.merge(articles, articles_for_sentiment[['Date', 'sentiment_label', 'sentiment_score']], on='Date', how='left')

    articles["sentiment_label"] = articles["sentiment_label"].fillna("neutral") # Fill NaN for articles without sentiment
    articles["sentiment_score"] = articles["sentiment_score"].fillna(0.5) # Fill NaN for articles without sentiment


    if ticker:
        # Save to cache WITH the index
        articles.to_csv(cache_path, index=False) # Save without index, as Date is a column
        logger.info(f"Saved sentiment data to {cache_path}")
    return articles